###################################################################################################
# File: TwinsUKReplication.py
# Version: 0.0
# Date: 06.01.2020
# Noam Bar, noam.bar@weizmann.ac.il
#
#
# Python version: 3.5
###################################################################################################

import sys
import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import explained_variance_score, r2_score
import numpy as np


def normalize_metabolomics(df):
    # log10 transform
    df = df.apply(np.log10)
    # robust standardization
    df = df.apply(lambda x: (x-x.median())/x.std())
    # impute missing values with minimum
    df = df.fillna(df.min())
    return df


def compute_replication(real_values, predicted_values):
    # find matching indexes
    overlap_idx = list(set(real_values.index).intersection(set(predicted_values.index)))
    overlap_cls = list(set(real_values.columns).intersection(set(predicted_values.columns)))

    real_values = real_values.loc[overlap_idx, overlap_cls]
    predicted_values = predicted_values.loc[overlap_idx, overlap_cls]

    results_df = pd.DataFrame(index=overlap_cls, columns=['pearson_r', 'pearson_p', 'spearman_r', 'spearman_p',
                                                          'r2_score', 'explained_variance_score', 'n_samples'])
    results_df.loc[:, 'n_samples'] = len(overlap_idx)
    for metab in overlap_cls:
        # pearson
        r, p = pearsonr(real_values[metab], predicted_values[metab])
        results_df.loc[metab, 'pearson_r'] = r
        results_df.loc[metab, 'pearson_p'] = p
        # spearman
        r, p = spearmanr(real_values[metab], predicted_values[metab])
        results_df.loc[metab, 'spearman_r'] = r
        results_df.loc[metab, 'spearman_p'] = p
        # coefficient of determination
        results_df.loc[metab, 'r2_score'] = r2_score(y_true=real_values[metab], y_pred=predicted_values[metab])
        # explained variance score
        results_df.loc[metab, 'explained_variance_score'] = explained_variance_score(y_true=real_values[metab],
                                                                                     y_pred=predicted_values[metab])
    return results_df


def main():
    # print (sys.argv[1], sys.argv[2], sys.argv[3])
    real_values = pd.read_csv(sys.argv[1], index_col=0).astype(float)
    real_values.columns = real_values.columns.astype(str)
    if sys.argv[2].endswith('csv'):
        predicted_values = pd.read_csv(sys.argv[2], index_col=0).astype(float)
    elif sys.argv[2].endswith('pkl'):
        predicted_values = pd.read_pickle(sys.argv[2]).astype(float)
    predicted_values.columns = predicted_values.columns.astype(str)
    output_path = sys.argv[3]

    real_values = normalize_metabolomics(real_values)
    results = compute_replication(real_values, predicted_values)
    results.to_csv(output_path)
    return

main()


