###################################################################################################
# File: CVPredictor_and_SHAP.py
# Version: 0.0
# Date: 04.06.2019
# Noam Bar, noam.bar@weizmann.ac.il
#
#
# Python version: 2.7
###################################################################################################

from __future__ import print_function
from sklearn import metrics
from AnalysisHelperFunctions import make_dir_if_not_exists, compute_abs_SHAP, compute_signed_abs_SHAP, log_run_details
import lightgbm as lgb
import shap
import os
import pickle
from scipy.stats.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, precision_recall_curve, explained_variance_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.linear_model import Lasso
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import sys

# LightGBM params
learning_rate = [0.1, 0.05, 0.02, 0.015, 0.01, 0.0075, 0.005, 0.002, 0.001, 0.0005, 0.0001]
num_leaves = range(2, 35)
max_depth = [-1, 2, 3, 4, 5, 10, 20, 40, 50]
min_data_in_leaf = range(1, 45, 2)
feature_fraction = [i / 10. for i in range(2, 11)]  # [1] when using dummy variables
metric = ['l2']
early_stopping_rounds = [None]
num_threads = [1]
verbose = [-1]
silent = [True]
n_estimators = range(100, 2500, 50)
bagging_fraction = [i / 10. for i in range(2, 11)]
bagging_freq = [0, 1, 2]
lambda_l1 = [0, 0.001, 0.005, 0.01, 0.1]

# Lasso params
alpha = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1]

lightgbm_rscv_space = {'learning_rate': learning_rate, 'max_depth': max_depth,
                       'feature_fraction': feature_fraction, 'num_leaves': num_leaves,
                       'min_data_in_leaf': min_data_in_leaf, 'metric': metric,
                       'early_stopping_rounds': early_stopping_rounds, 'n_estimators': n_estimators,
                       'bagging_fraction': bagging_fraction, 'bagging_freq': bagging_freq,
                       'num_threads': num_threads, 'verbose': verbose, 'silent': silent, 'lambda_l1': lambda_l1}

lasso_rscv_space = {'alpha': alpha}
rscv_space_dic = {'lightgbm': lightgbm_rscv_space, 'Lasso': lasso_rscv_space}
predictors_type = {'trees': ['lightgbm'], 'linear': ['Lasso']}


def randomized_search_cv_and_shap(command_args, Y, idx):
    # print('randomized_search_cv_and_shap', str(idx))
    if command_args.path_to_X.endswith('.csv'):
        X = pd.read_csv(command_args.path_to_X, index_col=0)
    else:
        X = pd.read_pickle(command_args.path_to_X)

    if command_args.log_transform_x:
        X = X.apply(np.log10)
    results_df = pd.DataFrame(index=Y.columns)
    predictions_df = pd.DataFrame(index=Y.index, columns=Y.columns)
    shap_values_dic = {}

    # if data is small, making this parameter smaller
    if X.shape[0] < 200 or Y.shape[0] < 200:
        min_data_in_leaf = range(1, 25, 2)
        lightgbm_rscv_space['min_data_in_leaf'] = min_data_in_leaf

    for y_name in Y.columns:

        y = Y[y_name].dropna().astype(float).copy()
        X_temp = X.loc[y.index].dropna(how='all').copy()
        y = y.loc[X_temp.index]

        if y.shape[0] < 200:
            min_data_in_leaf = range(1, 25, 2)
            lightgbm_rscv_space['min_data_in_leaf'] = min_data_in_leaf

        print (y_name)
        print (y.sort_values().unique())
        print(y.sort_values().unique() == np.array([0., 1.]))
        if (y.unique().shape[0] == 2) and ((type(y.unique().max()) == str) |
                                           (y.sort_values().unique() == np.array([0., 1.])).all()):
            classification_problem = True
        else:
            classification_problem = False

        groups = np.array(range(X_temp.shape[0]))
        group_kfold = GroupKFold(n_splits=command_args.k_folds)
        shap_values = pd.DataFrame(np.nan, index=X_temp.index, columns=X_temp.columns)
        final_pred = pd.DataFrame(index=X_temp.index, columns=[y_name])

        try:
            for train_index, test_index in group_kfold.split(X_temp, y, groups):
                X_train, X_test = X_temp.iloc[train_index, :], X_temp.iloc[test_index, :]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                model = _choose_model(command_args.model, classification_problem)

                rscv = RandomizedSearchCV(model, rscv_space_dic[command_args.model], n_iter=command_args.n_random)
                rscv.fit(X_train, y_train)
                # use best predictor according to random hyper-parameter search
                if classification_problem:
                    y_pred = rscv.best_estimator_.predict_proba(X_test)
                    final_pred.loc[X_test.index, :] = np.expand_dims(y_pred[:, 1], 1)
                else:
                    y_pred = rscv.best_estimator_.predict(X_test)
                    final_pred.loc[X_test.index, :] = np.expand_dims(y_pred, 1)
                # run SHAP
                if command_args.model in predictors_type['trees']:
                    explainer = shap.TreeExplainer(rscv.best_estimator_)
                    try:
                        # last column is the bias column
                        shap_values.loc[X_test.index, :] = explainer.shap_values(X_test)
                    except:
                        shap_values.loc[X_test.index, :] = explainer.shap_values(X_test)[:, :-1]
            shap_values_dic[y_name] = shap_values
            results_df = _evaluate_performance(y_name, final_pred.values.ravel(), y, results_df, classification_problem)
            predictions_df.loc[final_pred.index, y_name] = final_pred.values.ravel()
        except:
            print('RandomizedSearchCV failed with metabolite %s' % y_name)
            print ('y shape', y.shape)
            continue
    _save_temporary_files(command_args, idx, shap_values_dic, results_df, predictions_df)
    return


def _choose_model(model, classification_problem):
    if model == 'lightgbm':
        if classification_problem:
            return lgb.LGBMClassifier()
        else:
            return lgb.LGBMRegressor()
    elif model == 'Lasso':
        return Lasso()
    else:
        return None


def _save_temporary_files(command_args, idx, shap_values_dic, results_df, predictions_df):
    with open(command_args.output_dir + '/temp_resdf_' + str(idx) + '.pkl', 'wb') as fout:
        pickle.dump(results_df, fout)
    if command_args.model in predictors_type['trees']:
        with open(command_args.output_dir + '/temp_shap_' + str(idx) + '.pkl', 'wb') as fout:
            pickle.dump(shap_values_dic, fout)
    with open(command_args.output_dir + '/temp_pred_' + str(idx) + '.pkl', 'wb') as fout:
        pickle.dump(predictions_df, fout)
    return


def _evaluate_performance(y_name, y_pred, y_test, results_df, classification_problem):
    results_df.loc[y_name, 'Size'] = y_pred.shape[0]
    if classification_problem:
        # Prevalence
        results_df.loc[y_name, 'prevalence'] = float(y_test.sum()) / y_test.shape[0]
        # AUC
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
        results_df.loc[y_name, 'AUC'] = metrics.auc(fpr, tpr)
        # PR
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        results_df.loc[y_name, 'Precision_Recall'] = metrics.auc(recall, precision)
    else:
        results_df.loc[y_name, 'Coefficient_of_determination'] = r2_score(y_true=y_test, y_pred=y_pred)
        results_df.loc[y_name, 'explained_variance_score'] = explained_variance_score(y_true=y_test, y_pred=y_pred)
        results_df.loc[y_name, 'pearson_r'], results_df.loc[y_name, 'pearson_p'] = pearsonr(y_pred, y_test)
        results_df.loc[y_name, 'spearman_r'], results_df.loc[y_name, 'spearman_p'] = spearmanr(y_pred, y_test)
    return results_df


def concat_outputs(command_args):
    all_temp_files = os.listdir(command_args.output_dir)
    resdf_files = [command_args.output_dir + f for f in all_temp_files if f.startswith('temp_resdf_')]
    shap_files = [command_args.output_dir + f for f in all_temp_files if f.startswith('temp_shap_')]
    pred_files = [command_args.output_dir + f for f in all_temp_files if f.startswith('temp_pred_')]
    _concat_files(resdf_files, command_args.output_dir + '/results.pkl', how='dataframe')
    if command_args.model in predictors_type['trees']:
        _concat_files(shap_files, command_args.output_dir + '/shap_values.pkl', how='dic')
    _concat_files(pred_files, command_args.output_dir + '/predictions_df.pkl', how='dataframe', axis=1)
    return


def _concat_files(files, final_path, how='dataframe', axis=0):
    if how == 'dataframe':
        final_file = pd.DataFrame()
        for f in files:
            final_file = pd.concat((final_file, pd.read_pickle(f)), axis=axis)
            os.remove(f)
        with open(final_path, 'wb') as fout:
            pickle.dump(final_file, fout)
        final_file.to_csv(((final_path.split('.pkl')[0]).split('.dat')[0]) + '.csv')
    elif how == 'dic':
        final_file = {}
        for f in files:
            final_file.update(pd.read_pickle(f))
            os.remove(f)
        with open(final_path, 'wb') as fout:
            pickle.dump(final_file, fout)
    return


def _compute_abs_and_sign_SHAP(shap_dir, features_path):
    print('loading shap_values.pkl ...')
    shap_values_dic = pd.read_pickle(shap_dir + 'shap_values.pkl')
    if not os.path.exists(shap_dir + 'abs_shap.dat'):
        abs_shap = compute_abs_SHAP(shap_values_dic).astype(float)
        abs_shap.to_pickle(shap_dir + 'abs_shap.dat')
        abs_shap.to_csv(shap_dir + 'abs_shap.csv')
    else:
        abs_shap = pd.read_pickle(shap_dir + 'abs_shap.dat')

    if not os.path.exists(shap_dir + 'signed_shap.dat'):
        if features_path.endswith('.csv'):
            X = pd.read_csv(features_path, index_col=0)
        else:
            X = pd.read_pickle(features_path)

        signed_shap = compute_signed_abs_SHAP({ind: shap_values_dic[ind] for ind in abs_shap.index}, X).fillna(0)
        signed_shap.to_csv(shap_dir + 'signed_shap.csv')
    else:
        signed_shap = pd.read_pickle(shap_dir + 'signed_shap.dat')

    abs_signed_shap = (abs_shap.copy() * signed_shap.copy()).fillna(0)
    abs_signed_shap.to_csv(shap_dir + 'abs_signed_shap.csv')

    return

def upload_these_jobs(command_args):
    waiton = []
    if command_args.path_to_Y.endswith('.csv'):
        Y = pd.read_csv(command_args.path_to_Y, index_col=0)
    else:
        Y = pd.read_pickle(command_args.path_to_Y)

    for idx in range(0, Y.shape[1], command_args.n_cols_per_job):
        randomized_search_cv_and_shap(command_args, Y.iloc[:, idx:idx + command_args.n_cols_per_job], idx)

    # merge the temp results files
    concat_outputs(command_args)
    # compute absolute SHAP values and signed absolute SHAP values
    if command_args.model in predictors_type['trees']:
        _compute_abs_and_sign_SHAP(command_args.output_dir, command_args.path_to_X)
    return res


def main():
    print('main')
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', help='Path to output directory', type=str, default=None)
    parser.add_argument('-model', help='Which prediction model to use', type=str, default='lightgbm')
    parser.add_argument('-n_cols_per_job', help='Number of columns per job', type=int, default=10)
    parser.add_argument('-n_random', help='Number of random samples', type=int, default=20)
    parser.add_argument('-path_to_X', '--path_to_X', help='Path to features data - X', type=str, default='')
    parser.add_argument('-path_to_Y', help='Path to labels - Y', type=str, default='')
    parser.add_argument('-k_folds', help='Number of folds', type=int, default=10)
    parser.add_argument('-only_concat', help='Whether to only concatenate the output files', type=bool, default=False)
    parser.add_argument('-only_compute_abs_SHAP', help='Whether to only compute absolute SHAP values', type=bool,
                        default=False)
    parser.add_argument('-log_transform_x', help='Whether to log transform the X', type=bool, default=False)
    command_args = parser.parse_args()

    if (not os.path.exists(command_args.path_to_X)) or (not os.path.exists(command_args.path_to_Y)):
        print("X or Y doesn't exist!")
        return
    if command_args.n_cols_per_job < 1 or command_args.n_cols_per_job > 1000:
        print("n_cols_per_job must be between 1 and 1000")
        return

    if command_args.only_concat:
        concat_outputs(command_args)
        return

    if command_args.only_compute_abs_SHAP:
        _compute_abs_and_sign_SHAP(command_args.output_dir, command_args.path_to_X);
        return

    make_dir_if_not_exists(command_args.output_dir)
    log_run_details(command_args)
    upload_these_jobs(command_args)


if __name__ == "__main__":
    main()
