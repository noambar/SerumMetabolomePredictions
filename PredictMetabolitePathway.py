###################################################################################################
# File: PredictMetaboliteFunction.py
# Version: 0.0
# Date: 23.02.2020
# Noam Bar, noam.bar@weizmann.ac.il
#
#
# Python version: 2.7
###################################################################################################

from __future__ import print_function
from Analyses.AnalysisHelperFunctions import make_dir_if_not_exists, compute_abs_SHAP, compute_signed_abs_SHAP, \
    log_run_details, balance_sampleing
import lightgbm as lgb
import os
import pickle
import argparse
import numpy as np
import pandas as pd
from addloglevels import sethandlers
from queue.qp import qp, fakeqp
from datetime import datetime

from sklearn.model_selection import train_test_split

params = {
          'objective': 'multiclass',
          'num_leaves': 25,
          'max_depth': 4,
          'learning_rate': 0.005,
          'bagging_fraction': 0.8,  # subsample
          'feature_fraction': 0.8,  # colsample_bytree
          'bagging_freq': 1,        # subsample_freq
          'bagging_seed': 2018,
          'verbosity': -1,
          'silent': True,
          'num_threads': 2,
          'class_weight': 'balanced'}


def predict_pathway(command_args, idx):
    # read the data
    Y = pd.read_csv(command_args.path_to_Y, index_col=0)
    Y.index = Y.index.astype(str)
    num_class = Y.iloc[:, 0].unique().shape[0]
    params.update({'n_estimators': command_args.ntrees, 'early_stopping_rounds': command_args.early_stopping_rounds})

    Xs = [pd.read_csv(x, index_col=0) for x in command_args.path_to_X]

    y_names = Y.iloc[idx:idx+command_args.n_cols_per_job].index

    for X, name in zip(Xs, command_args.names):
        results = pd.DataFrame(index=y_names, columns=range(num_class))
        X = X.loc[Y.index].copy()
        # leave one out for the relevant sample
        for y_name in y_names:
            # drop the y_name
            temp_X = X.drop(y_name).copy()
            temp_y = Y.drop(y_name).values.ravel()
            X_train, X_val, y_train, y_val = train_test_split(temp_X, temp_y, test_size=command_args.val_size)

            if command_args.over_sample:
                y_train = balance_sampleing(y_train)
                X_train = X_train.loc[y_train.index]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

            # obtain a prediction for the metabolite
            results.loc[y_name, :] = model.predict_proba(X.loc[[y_name]])
        _save_temporary_files(results, command_args, idx, name)


def _save_temporary_files(df, command_args, idx, name):
    with open(command_args.output_dir + '/temp___' + name + '___' + str(idx) + '.pkl', 'wb') as fout:
        pickle.dump(df, fout)
    return


def concat_outputs(command_args):
    print('concat_outputs')
    all_temp_files = [f for f in os.listdir(command_args.output_dir) if f.startswith('temp___')]
    names = set([s.split('___')[1] for s in all_temp_files])
    for name in names:
        files = [command_args.output_dir + f for f in all_temp_files if f.startswith('temp___' + name)]
        _concat_files(files, os.path.join(command_args.output_dir, name + '.pkl'), how='dataframe')
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

def predict_test(command_args):
    """

    :param command_args:
    :return:
    """
    Y = pd.read_csv(command_args.path_to_Y, index_col=0)
    Y.index = Y.index.astype(str)
    num_class = Y.iloc[:, 0].unique().shape[0]
    params.update({'n_estimators': command_args.ntrees, 'early_stopping_rounds': command_args.early_stopping_rounds})

    Xs = [pd.read_csv(x, index_col=0) for x in command_args.path_to_X]
    for X, name in zip(Xs, command_args.names):
        X_known = X.loc[Y.index].copy()
        X_unknown = X.loc[~X.index.isin(X_known.index)].copy()
        results = pd.DataFrame(index=X_unknown.index, columns=range(num_class))

        X_train, X_val, y_train, y_val = train_test_split(X_known, Y.values.ravel(), test_size=command_args.val_size)

        if command_args.over_sample:
            y_train = balance_sampleing(y_train)
            X_train = X_train.loc[y_train.index]

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

        # obtain a prediction for the metabolites
        results.loc[X_unknown.index, :] = model.predict_proba(X_unknown)
        results.to_csv(os.path.join(command_args.output_dir, 'unknowns_' + name + '.csv'))
    return

def upload_these_jobs(q, command_args):
    """

    :param q:
    :param command_args:
    :return:
    """
    print('upload_these_jobs')
    waiton = []
    Y = pd.read_csv(command_args.path_to_Y, index_col=0)

    for idx in range(0, Y.shape[0], command_args.n_cols_per_job):
        waiton.append(q.method(predict_pathway, (command_args, idx)))
    print('Will run a total of ' + str(len(waiton)) + ' jobs')
    res = q.waitforresults(waiton)
    # merge the temp results files
    concat_outputs(command_args)

    predict_test(command_args)
    return res


def _convert_comma_separated_to_list(s):
    return s.split(',')


def main():
    print('main')
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', help='Path to output directory', type=str, default=None)
    parser.add_argument('-n_cols_per_job', help='Number of columns per job', type=int, default=10)
    parser.add_argument('-path_to_X', '--path_to_X', help='Path to features data - X, separated by comma', type=str,
                        default='/net/mraid08/export/jafar/Microbiome/Analyses/Noamba/Metabolon/Paper_v4/unknown_pathway_prediction/metabolomics_levels_mean_null.csv')
    parser.add_argument('-path_to_Y', help='Path to labels - Y', type=str,
                        default='/net/mraid08/export/jafar/Microbiome/Analyses/Noamba/Metabolon/Paper_v4/unknown_pathway_prediction/super_pathway_y.csv')
    parser.add_argument('-names', help='names of Xs, separated by comma. Must be same length as the number of Xs.',
                        type=str, default='levels')
    parser.add_argument('-ntrees', help='The number of trees for training', type=int, default=2000)
    parser.add_argument('-val_size', help='The fraction of validation for the training', type=float, default=0.2)
    parser.add_argument('-early_stopping_rounds', help='The number of early stopping rounds for the training',
                        type=int, default=50)
    parser.add_argument('-over_sample', help='Whether to over sample the samples', type=bool, default=False)
    parser.add_argument('-only_concat', help='Whether to only concatenate the output files', type=bool, default=False)
    parser.add_argument('-only_predict_test', help='Whether to only run the predict_test function', type=bool, default=False)
    parser.add_argument('-mem_def', help='Amount of memory per job', type=int, default=1)
    parser.add_argument('-job_name', help='Job preffix for q', type=str, default='PathwayClassifier')
    command_args = parser.parse_args()

    command_args.path_to_X = _convert_comma_separated_to_list(command_args.path_to_X)
    for x in command_args.path_to_X:
        if not os.path.exists(x):
            print (x, 'does not exist')
            return
    if not os.path.exists(command_args.path_to_Y):
        print(command_args.path_to_Y, 'does not exist!')
        return

    command_args.names = _convert_comma_separated_to_list(command_args.names)
    assert len(command_args.names) == len(command_args.path_to_X)

    if command_args.n_cols_per_job < 1 or command_args.n_cols_per_job > 1000:
        print("n_cols_per_job must be between 1 and 1000")
        return

    if command_args.only_concat:
        concat_outputs(command_args)
        return

    if command_args.only_predict_test:
        predict_test(command_args)
        return

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