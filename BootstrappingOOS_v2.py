###################################################################################################
# File: BootstrappingOOS_v2.py
# Version: 0.0
# Date: 09.02.2020
# Noam Bar, noam.bar@weizmann.ac.il
#
#
# Python version: 2.7
###################################################################################################

from __future__ import print_function
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from Analyses.AnalysisHelperFunctions import make_dir_if_not_exists, log_run_details
import lightgbm as lgb
# import shap
import os
import pickle
from scipy.stats.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, precision_recall_curve, explained_variance_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Lasso
import argparse
import numpy as np
import pandas as pd
from addloglevels import sethandlers
from queue.qp import qp, fakeqp
from datetime import datetime
# import Utils
import sys
from sklearn.utils import resample

import warnings

warnings.filterwarnings('ignore')

# constant hyper parameters for all runs and permutations
PREDICTOR_PARAMS_DEFAULT = {'learning_rate': 0.01, 'max_depth': 5,
                            'feature_fraction': 0.8, 'num_leaves': 25,
                            'min_data_in_leaf': 15, 'metric': 'l2',
                            'early_stopping_rounds': None, 'n_estimators': 200,
                            'bagging_fraction': 0.9, 'bagging_freq': 5,
                            'num_threads': 1, 'verbose': -1, 'silent': True}

# constant hyper parameters for all runs and permutations - Suggested by Eran 5.12.2018
PREDICTOR_PARAMS_MULTI_FEATURES = {'learning_rate': 0.005,
                                   'feature_fraction': 0.2,
                                   'min_data_in_leaf': 15, 'metric': 'l2',
                                   'early_stopping_rounds': None, 'n_estimators': 2000,
                                   'bagging_fraction': 0.8, 'bagging_freq': 1,
                                   'num_threads': 1, 'verbose': -1, 'silent': True}

LASSO_PARAMS = {'alpha' : 0.5}

supported_prediction_models = ['lightgbm', 'Lasso']

REGRESSION_EVAL_METHODS = ['pearson_r', 'Coefficient_of_determination', 'explained_variance_score']
CLASSIFICATION_EVAL_METHODS = ['AUC', 'Precision_Recall']


# keep all bootstrap results, think of a smart format to do that.
# start using csv instead of pickle
# try and make code compatible with python3 - queue


def _cross_validation(X, y, y_name, do_bootstrap, command_args, model, classification_problem,
                      random_state=0):
    """

    :param X:
    :param y:
    :param y_name:
    :param do_bootstrap:
    :param command_args:
    :param classification_problem:
    :param predictor_params:
    :param random_state:
    :return:
    """
    groups = np.array(range(X.shape[0]))
    group_kfold = GroupKFold(n_splits=command_args.k_folds)
    final_pred = pd.DataFrame(index=X.index, columns=[y_name])
    for train_index, test_index in group_kfold.split(X, y, groups):
        X_train, X_test = X.iloc[train_index, :].copy(), X.iloc[test_index, :].copy()
        y_train, y_test = y.iloc[train_index].copy(), y.iloc[test_index].copy()
        if do_bootstrap:
            boot = resample(X_train.index, replace=True, n_samples=X_train.shape[0])  # , random_state=random_state
            X_train, y_train = X_train.loc[boot], y_train.loc[boot]

        # if classification_problem:
        #     gbm = lgb.LGBMClassifier(**predictor_params)
        # else:
        #     gbm = lgb.LGBMRegressor(**predictor_params)
        model.fit(X_train, y_train)
        if classification_problem:
            y_pred = model.predict_proba(X_test)
            final_pred.loc[X_test.index, :] = np.expand_dims(y_pred[:, 1], 1)
        else:
            y_pred = model.predict(X_test)
            final_pred.loc[X_test.index, :] = np.expand_dims(y_pred, 1)
        final_pred.loc[X_test.index, :] = np.expand_dims(y_pred, 1)
    return final_pred

def _choose_model(model, classification_problem, predictor_params):
    if model == 'lightgbm':
        if classification_problem:
            return lgb.LGBMClassifier(**predictor_params)
        else:
            return lgb.LGBMRegressor(**predictor_params)
    # In a linear regression model we don't allow any missing values,
    # so I need to add code that checks no missing values are present
    elif model == 'Lasso':
        return Lasso(alpha=5e-4)
    else:
        return None


def perform_n_bootstraps(command_args, y_name):
    """

    :param command_args:
    :param y_name:
    :return:
    """
    if command_args.path_to_Y.endswith('.csv'):
        y = pd.read_csv(command_args.path_to_Y, index_col=0)[y_name].dropna()
    else:
        y = pd.read_pickle(command_args.path_to_Y)[y_name].dropna()
    Xs = []
    for x in command_args.Xs:
        if x.endswith('.csv'):
            Xs.append(pd.read_csv(x, index_col=0))
        else:
            Xs.append(pd.read_pickle(x))
    X = pd.concat(Xs, axis=1, sort=True)
    # remove rows from X which are all missing
    X = X.loc[y.index].dropna(how='all', axis=0)
    y = y.loc[X.index]

    if command_args.log_transform:
        X = X.apply(np.log10)

    true_results_df = pd.DataFrame(index=[y_name], columns=['Size', 'Coefficient_of_determination',
                                                            'explained_variance_score',
                                                            'pearson_r', 'pearson_p', 'spearman_r',
                                                            'spearman_p',
                                                            'prevalence', 'AUC', 'Precision_Recall'])
    # check if regression or classification problem
    if y.unique().shape[0] == 2:
        classification_problem = True
        eval_method = CLASSIFICATION_EVAL_METHODS
    else:
        classification_problem = False
        eval_method = REGRESSION_EVAL_METHODS

    predictor_params = PREDICTOR_PARAMS_DEFAULT
    if command_args.multi_features:
        predictor_params = PREDICTOR_PARAMS_MULTI_FEATURES

    # define model for prediction
    model = _choose_model(command_args.model, classification_problem, predictor_params)

    # run full data
    final_pred = _cross_validation(X, y, y_name, do_bootstrap=False, command_args=command_args,
                                   classification_problem=classification_problem, model=model)
    true_results_df = evaluate_performance(y_name, final_pred.values.ravel(), y, true_results_df, classification_problem)

    # If real data estimate is lower than 0, don't bootstrap
    if (not command_args.bootstrap_negative_estimate) and (true_results_df.loc[y_name, eval_method[0]] < 0):
        aggr_results = {k: [None for i in range(command_args.n_BS)] for k in eval_method}
        return aggr_results, true_results_df

    # run n_BS bootstraps
    aggr_results = {k:[] for k in eval_method}
    for i in range(command_args.n_BS):
        if command_args.rand_folds:  # randomize the folds
            rand_idx = np.random.permutation(X.index)
            X, y = X.loc[rand_idx], y.loc[rand_idx]
        final_pred = _cross_validation(X, y, y_name, do_bootstrap=True, command_args=command_args,
                                       classification_problem=classification_problem, model=model,
                                       random_state=i)
        results_df = evaluate_performance(y_name, final_pred.values.ravel(), y, pd.DataFrame(), classification_problem)

        for k in eval_method:
            aggr_results[k].append(results_df.loc[y_name, k])

    return aggr_results, true_results_df


def evaluate_performance(y_name, y_pred, y_test, results_df, classification_problem):
    """

    :param y_name:
    :param y_pred:
    :param y_test:
    :param results_df:
    :param classification_problem:
    :return:
    """
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


def oos_bootstrap_per_y(command_args, y_name):
    """

    :param command_args:
    :param y_name:
    :return:
    """
    # run the bootstrapping
    aggr_perm, true_results_df = perform_n_bootstraps(command_args, y_name)
    # save bootstrapping results
    for k in aggr_perm:
        bootstrap_estimates_df = pd.DataFrame(index=[y_name], columns=range(command_args.n_BS))
        bootstrap_estimates_df.loc[y_name, :] = aggr_perm[k]
        pd.to_pickle(bootstrap_estimates_df, command_args.output_dir + 'temp_' + k + '_' + y_name + '.dat')
    # save real estimate
    pd.to_pickle(true_results_df, command_args.output_dir + 'temp_estimate_' + y_name + '.dat')
    return


def upload_job_per_y(q, command_args):
    """

    :param q:
    :param command_args:
    :return:
    """
    print('upload_job_per_y')
    waiton = []
    if command_args.path_to_Y.endswith('.csv'):
        y = pd.read_csv(command_args.path_to_Y, index_col=0)
    else:
        y = pd.read_pickle(command_args.path_to_Y)

    for y_name in y.columns:
        if not os.path.exists(command_args.output_dir + 'temp_estimate_' + y_name + '.dat'):
            waiton.append(q.method(oos_bootstrap_per_y, (command_args,
                                                         y_name)
                                   ))
    print('Will run a total of ' + str(len(waiton)) + ' jobs')
    res = q.waitforresults(waiton)
    # merge the temp results files
    concat_outputs(command_args)
    return res


def concat_outputs(command_args):
    """

    :param command_args:
    :return:
    """
    print('concat_outputs')
    all_temp_files = os.listdir(command_args.output_dir)
    est_files = [command_args.output_dir + f for f in all_temp_files if f.startswith('temp_estimate_')]
    if len(est_files) > 0:
        _concat_files(est_files, command_args.output_dir + '/estimates.pkl', how='dataframe')
    for k in REGRESSION_EVAL_METHODS + CLASSIFICATION_EVAL_METHODS:
        bs_files = [command_args.output_dir + f for f in all_temp_files if f.startswith('temp_' + k)]
        if len(bs_files) > 0:
            _concat_files(bs_files, os.path.join(command_args.output_dir, k + '.pkl'), how='dataframe')
    return


def _concat_files(files, final_path, how='dataframe', axis=0):
    """

    :param files:
    :param final_path:
    :param how:
    :param axis:
    :return:
    """
    if how == 'dataframe':
        final_file = pd.DataFrame()
        for f in files:
            final_file = pd.concat((final_file, pd.read_pickle(f)), axis=axis)
            os.remove(f)
        # with open(final_path, 'wb') as fout:
            # pickle.dump(final_file, fout)
        final_file.to_csv(((final_path.split('.pkl')[0]).split('.dat')[0]) + '.csv')
    elif how == 'dic':
        final_file = {}
        for f in files:
            final_file.update(pd.read_pickle(f))
            os.remove(f)
        with open(final_path, 'wb') as fout:
            pickle.dump(final_file, fout)
    return

def _convert_comma_separated_to_list(s):
    return s.split(',')


def main():
    """

    :return:
    """
    print('main')
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', help='Path to output directory', type=str, default=None)
    parser.add_argument('-path_to_X', '--path_to_X', help='Path to features data - X', type=str,
                        default='/home/noamba/Analyses/Noamba/Metabolon/SHAP/dataframes/mar17_phenome.dat')
    parser.add_argument('-path_to_Y', help='Path to labels - Y', type=str,
                        default='/home/noamba/Analyses/Noamba/Metabolon/technical_noise/dataframes/mar17_metabolomics_grouped085_unnormed_fillna_min_dayfromfirstsample_regressed_rzs.dat')
    # default='/home/noamba/Analyses/Noamba/Metabolon/SHAP/dataframes/mar17_metabolomics_grouped085_unnormed_fillna_min_dayfromfirstsample_regressed_rzs.dat')
    parser.add_argument('-model', help='Which prediction model to use', type=str, default='lightgbm')
    parser.add_argument('-k_folds', help='Number of folds', type=int, default=10)
    parser.add_argument('-only_concat', help='Whether to only concatenate the output files', type=bool, default=False)
    parser.add_argument('-mem_def', help='Number of folds', type=int, default=1)
    parser.add_argument('-n_BS', help='Number of CI permutations', type=int, default=1000)
    parser.add_argument('-multi_features', help='Whether to use the set of hyper parameters designed for a large '
                                                'number of features', type=bool, default=False)
    parser.add_argument('-bootstrap_negative_estimate', help='Whether to run bootstrapping on estimates which are '
                                                             'negative', type=bool, default=False)
    parser.add_argument('-log_transform', help='Whether to log transform the data', type=bool, default=False)
    parser.add_argument('-rand_folds', help='Whether to randomize the folds when bootstrapping', type=bool,
                        default=False)
    parser.add_argument('-job_name', help='Job preffix for q', type=str, default='')

    command_args = parser.parse_args()
    # check X and y exist

    command_args.Xs = _convert_comma_separated_to_list(command_args.path_to_X)

    for x in command_args.Xs:
        if not os.path.exists(x):
            print (x, 'does not exist.')
            return

    if not os.path.exists(command_args.path_to_Y):
        print("Y doesn't exist!")
        return
    # check model is legal
    if command_args.model not in supported_prediction_models:
        print('chosen model must be one of:', ', '.join(supported_prediction_models))
        return
    # only concat results, do not run bootstrapping
    if command_args.only_concat:
        concat_outputs(command_args)
        return

    make_dir_if_not_exists(command_args.output_dir)

    log_run_details(command_args)

    if len(command_args.job_name) == 0:
        job_name = 'bs_' + command_args.output_dir.split('/')[-2]
    else:
        job_name = command_args.job_name
    # with fakeqp(jobname = job_name, q=['himem7.q'], mem_def = '1G',
    with qp(jobname=job_name, q=['himem7.q'], mem_def='1G',
            trds_def=2, tryrerun=True, max_u=550, delay_batch=5, max_r=550) as q:
        os.chdir("/net/mraid08/export/jafar/Microbiome/Analyses/Noamba/temp_q_dir/")
        q.startpermanentrun()
        upload_job_per_y(q, command_args)


if __name__ == "__main__":
    sethandlers()
    main()