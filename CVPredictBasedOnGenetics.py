###################################################################################################
# File: CVPredictBasedOnGenetics.py
# Version: 0.0
# Date: 30.12.2019
# Noam Bar, noam.bar@weizmann.ac.il
#
#
# Python version: 2.7
###################################################################################################

# for a given y:
# run in CV, where in every fold:
# 1. run plink and treat y as quantitative, provide some p-value threshold.
# 2. use plink to extract the associated SNPs.
# 3. build a regression model based on these SNPs and predict the test.
# 4. aggregate results and compute the overall r2.

from __future__ import print_function
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from Analyses.AnalysisHelperFunctions import make_dir_if_not_exists, compute_abs_SHAP, compute_signed_abs_SHAP, \
    log_run_details
import lightgbm as lgb
import shap
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
from shutil import copyfile
# import Utils # TODO: change all pickle writings to csv
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

plink = '/home/noamba/Genie/Bin/plink1/plink'

def copy_plink_files(origin_basename, destination_dir):
    make_dir_if_not_exists(destination_dir)
    os.chdir(destination_dir)
    # using the command line "cp"
    os.system(' '.join(['cp', origin_basename + '*', '.']))
    # # using python
    # basename = os.path.basename(origin_basename)
    # basedir = origin_basename.split(basename)[0]
    # for s in [s for s in os.listdir(basedir) if s.startswith(basename)]:
    #     copyfile(os.path.join(basedir, s), os.path.join(destination_dir, s))
    return

def randomized_search_cv_and_plink(command_args, Y, idx):
    # create a new folder for this run
    destination_dir = os.path.join(command_args.output_dir, 'plink%0.1d'%idx)
    # copy the plink files into the directory and change the working directory
    copy_plink_files(command_args.path_to_plink_bfiles, destination_dir)
    # extract the samples for analysis and prepare the base fam file
    basename = os.path.basename(command_args.path_to_plink_bfiles)
    basic_fam_path = os.path.join(destination_dir, basename + '.fam')
    basic_fam = pd.read_csv(basic_fam_path, sep=' ', header=-1, index_col=0)
    basic_fam.index.names = ['RegistrationCode']
    basic_fam[[1, 2, 3, 4]] = basic_fam[[1, 2, 3, 4]].astype(int)
    X = basic_fam.copy()
    Y = Y.loc[X.index].dropna(how='all')
    X = X.loc[Y.index]

    results_df = pd.DataFrame(index=Y.columns)
    predictions_df = pd.DataFrame(index=Y.index, columns=Y.columns)

    # added on 5.3.2019 - if data is small, making this parameter smaller
    if X.shape[0] < 200 or Y.shape[0] < 200:
        min_data_in_leaf = range(1, 25, 2)
        lightgbm_rscv_space['min_data_in_leaf'] = min_data_in_leaf

    for y_name in Y.columns:
        basic_fam[5] = Y.loc[:, y_name].copy()
        basic_fam.to_csv(basic_fam_path, sep=' ', header=False)

        y = Y[y_name].dropna().astype(float).copy()
        # X_temp = X.loc[y.index].dropna(how='all').copy()
        # y = y.loc[X_temp.index]

        if y.shape[0] < 200:
            min_data_in_leaf = range(1, 25, 2)
            lightgbm_rscv_space['min_data_in_leaf'] = min_data_in_leaf

        if y.unique().shape[0] == 2:
            classification_problem = True
        else:
            classification_problem = False

        groups = np.array(range(y.shape[0]))
        group_kfold = GroupKFold(n_splits=command_args.k_folds)
        # shap_values = pd.DataFrame(np.nan, index=X_temp.index, columns=X_temp.columns)
        final_pred = pd.DataFrame(index=y.index, columns=[y_name])

        for train_index, test_index in group_kfold.split(X, y, groups):
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            train_ids, test_ids = y_train.index, y_test.index
            # save ids of train to temp file
            with open('keep.txt', 'w') as handle:
                for id in train_ids:
                    handle.write(str(id) + ' ' + str(id) + '\n')

            # use plink to extract samples from temp file
            os.system(' '.join([plink, '--bfile', basename, '--assoc', '--pfilter', str(command_args.pfilter),
                                '--out', y_name.replace(' ', '_'), '--adjust', '--keep', 'keep.txt']))
            # parse adjusted results
            snp_results = pd.read_csv(os.path.join(destination_dir, y_name.replace(' ', '_') + '.qassoc.adjusted'), header=0,
                                      sep='\s+')
            # keep only top 10 snps that pass Bonferonni
            snp_results = snp_results[snp_results['BONF'] < 0.05].head(10)
            if snp_results.shape[0] == 0:
                final_pred.loc[test_ids] = y_train.mean()
                continue
            # save snps to file
            snps_to_pull = snp_results['SNP'].tolist()
            snp_results['SNP'].to_csv('snps.txt', index=None)
            # create a temp X dataframe
            os.system(' '.join([plink, '--bfile', basename, '--extract', 'snps.txt', '--make-bed', '--out',
                                'temp_bed']))
            # convert to ped and mark snps as 1, 2
            os.system(' '.join([plink, '--bfile', 'temp_bed', '--recode12', '--tab', '--out', 'temp_ped']))
            # parse ped file to create X
            X_temp = pd.read_csv('temp_ped.ped', sep='\s+', header=None, index_col=0).loc[:, 6:].astype(float)
            X_temp.index.names = ['RegistrationCode']
            X_train, X_test = X_temp.loc[train_ids, :], X_temp.loc[test_ids, :]

            if X_train.drop_duplicates().shape[0] == 1:
                final_pred.loc[test_ids] = y_train.mean()
                continue

            model = _choose_model(command_args.model, classification_problem)

            rscv = RandomizedSearchCV(model, rscv_space_dic[command_args.model], n_iter=command_args.n_random)
            try:
                rscv.fit(X_train, y_train)
            except Exception:
                final_pred.loc[test_ids] = y_train.mean()
                continue
            # use best predictor according to random hyper-parameter search
            if classification_problem:
                y_pred = rscv.best_estimator_.predict_proba(X_test)
                final_pred.loc[X_test.index, :] = np.expand_dims(y_pred[:, 1], 1)
            else:
                y_pred = rscv.best_estimator_.predict(X_test)
                final_pred.loc[X_test.index, :] = np.expand_dims(y_pred, 1)

        results_df = _evaluate_performance(y_name, final_pred.values.ravel(), y, results_df, classification_problem)
        predictions_df.loc[final_pred.index, y_name] = final_pred.values.ravel()
    # remove temp files
    # for file2remove in os.listdir(destination_dir): # TODO: implement well, currently crashes because there is a directory in it
    #     os.unlink(os.path.join(destination_dir, file2remove))
        # except:
        #     print('RandomizedSearchCV failed with metabolite %s' % y_name, command_args.job_name)
        #     print ('y shape', y.shape)
        #     continue
    _save_temporary_files(command_args, idx, results_df, predictions_df)
    return



def _choose_model(model, classification_problem):
    if model == 'lightgbm':
        if classification_problem:
            return lgb.LGBMClassifier()
        else:
            return lgb.LGBMRegressor()
    # In a linear regression model we don't allow any missing values,
    # so I need to add code that checks no missing values are present
    elif model == 'Lasso':
        return Lasso()
    else:
        return None


def _save_temporary_files(command_args, idx, results_df, predictions_df):
    with open(command_args.output_dir + '/temp_resdf_' + str(idx) + '.pkl', 'wb') as fout:
        pickle.dump(results_df, fout)
    # if command_args.model in predictors_type['trees']:
    #     with open(command_args.output_dir + '/temp_shap_' + str(idx) + '.pkl', 'wb') as fout:
    #         pickle.dump(shap_values_dic, fout)
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
    print('concat_outputs')
    all_temp_files = os.listdir(command_args.output_dir)
    resdf_files = [command_args.output_dir + f for f in all_temp_files if f.startswith('temp_resdf_')]
    # shap_files = [command_args.output_dir + f for f in all_temp_files if f.startswith('temp_shap_')]
    pred_files = [command_args.output_dir + f for f in all_temp_files if f.startswith('temp_pred_')]
    _concat_files(resdf_files, command_args.output_dir + '/results.pkl', how='dataframe')
    # if command_args.model in predictors_type['trees']:
    #     _concat_files(shap_files, command_args.output_dir + '/shap_values.pkl', how='dic')
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
    # TODO: add also csv option...


def upload_these_jobs(q, command_args):
    print('upload_these_jobs')
    waiton = []
    if command_args.path_to_Y.endswith('.csv'):
        Y = pd.read_csv(command_args.path_to_Y, index_col=0)
    else:
        Y = pd.read_pickle(command_args.path_to_Y)

    for idx in range(0, Y.shape[1], command_args.n_cols_per_job):
        waiton.append(q.method(randomized_search_cv_and_plink, (command_args,
                                                               Y.iloc[:, idx:idx + command_args.n_cols_per_job],
                                                               idx)))
    print('Will run a total of ' + str(len(waiton)) + ' jobs')
    res = q.waitforresults(waiton)
    # merge the temp results files
    concat_outputs(command_args)
    # compute absolute SHAP values and signed absolute SHAP values
    # if command_args.model in predictors_type['trees']:
    #     _compute_abs_and_sign_SHAP(command_args.output_dir, command_args.path_to_X)
    return res


def main():
    print('main')
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', help='Path to output directory', type=str, default=None)
    parser.add_argument('-model', help='Which prediction model to use', type=str, default='lightgbm')
    parser.add_argument('-n_cols_per_job', help='Number of columns per job', type=int, default=2)
    parser.add_argument('-n_random', help='Number of random samples', type=int, default=20)
    parser.add_argument('-pfilter', help='Threshold for p-value in association test', type=float, default=0.00001)
    parser.add_argument('-path_to_plink_bfiles', '--path_to_plink_bfiles', help='Path to basename plink data', type=str,
                        default='/net/mraid08/export/jafar/Microbiome/Analyses/Noamba/Metabolon/Genetics/PNP_autosomal_clean2_nodfukim_norelated_Metabolon')
    parser.add_argument('-path_to_Y', help='Path to labels - Y', type=str,
                        default='/net/mraid08/export/jafar/Microbiome/Analyses/Noamba/Metabolon/technical_noise/dataframes/mar17_metabolomics_grouped085_unnormed_fillna_min_dayfromfirstsample_regressed_rzs_regid.csv')
    parser.add_argument('-k_folds', help='Number of folds', type=int, default=10)
    parser.add_argument('-only_concat', help='Whether to only concatenate the output files', type=bool, default=False)

    parser.add_argument('-mem_def', help='Amount of memory per job', type=int, default=2)
    parser.add_argument('-job_name', help='Job preffix for q', type=str, default='plink-cv')
    command_args = parser.parse_args()

    if (not os.path.exists(command_args.path_to_Y)): # (not os.path.exists(command_args.path_to_X)) or
        print("X or Y doesn't exist!")
        return
    if command_args.n_cols_per_job < 1 or command_args.n_cols_per_job > 1000:
        print("n_cols_per_job must be between 1 and 1000")
        return

    if command_args.only_concat:
        concat_outputs(command_args)
        return

    # if command_args.only_compute_abs_SHAP:
    #     _compute_abs_and_sign_SHAP(command_args.output_dir, command_args.path_to_X);
    #     return

    make_dir_if_not_exists(command_args.output_dir)

    log_run_details(command_args)

    # qp = fakeqp
    with qp(jobname=command_args.job_name, q=['himem7.q'], mem_def=str(command_args.mem_def) + 'G',
            trds_def=2, tryrerun=True, max_u=650, delay_batch=5) as q:
        os.chdir("/net/mraid08/export/jafar/Microbiome/Analyses/Noamba/temp_q_dir/")
        q.startpermanentrun()
        upload_these_jobs(q, command_args)


if __name__ == "__main__":
    sethandlers()
    main()