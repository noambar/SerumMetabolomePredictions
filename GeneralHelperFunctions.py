###################################################################################################
# File: GeneralHelperFunctions.py
# Version: 0.0
# Date: 09.07.2019
# Noam Bar, noam.bar@weizmann.ac.il
#
#
# Python version: 3.5
###################################################################################################

import pandas as pd
import numpy as np
import os
import re
import pickle
import random
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
import matplotlib as mpl
import seaborn as sns
import sys
import copy
import shutil

# try:
#     from stats.HigherLevelRanksum import directed_mannwhitneyu
# except:
#     from Utils.HigherLevelRanksum import directed_mannwhitneyu
from scipy.stats.stats import spearmanr, pearsonr, mannwhitneyu, distributions, find_repeats, rankdata, tiecorrect
from scipy.stats import ttest_ind, kstest, ks_2samp, shapiro, wilcoxon, binom_test, gmean, fisher_exact
from scipy.spatial.distance import braycurtis, pdist, squareform, euclidean, cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, GroupKFold, StratifiedShuffleSplit, RandomizedSearchCV, GridSearchCV

from sklearn.metrics import precision_recall_curve, mean_squared_error, r2_score, explained_variance_score,\
    roc_auc_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, log_loss,\
    accuracy_score

from sklearn import linear_model
from sklearn.linear_model import Lasso, LogisticRegression
from mne.stats.multi_comp import fdr_correction
# from skbio.diversity.alpha import shannon

import umap
import lightgbm as lgb
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
from tqdm import tnrange, tqdm_notebook
import tqdm
from time import sleep
from itertools import combinations
import statsmodels.api as sm
import statsmodels.tools.tools as sm_tools
from statsmodels.discrete.discrete_model import Logit

from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

# rcParams.update({'figure.autolayout': True})

def print_plus(i):
    print (str(datetime.now()) + str(i[0]))
    i[0] += 1


def clip_min_max(df):
    return df.clip(lower=df.quantile(2. / df.shape[0]), upper=df.quantile(float(df.shape[0] - 1) / df.shape[0]),
                   axis=1).copy()

def compute_moving_average(x, window_size=100, method=np.median):
    if window_size >= len(x):
        return [method(x)] * len(x)
    conv_vals = []
    for i in range(len(x)):
        conv_vals += [method(x[max(0, i - window_size // 2):i + window_size // 2])]
    return np.array(conv_vals)

def moving_average(x, N, mode='same'):
    assert mode in ['full', 'same', 'valid']
    return np.convolve(x, np.ones((N,))/N, mode=mode)


def BrayCurtis(X):
    return braycurtis(X[0, :], X[1:, ])


def read_fasta_to_dict(path):
    return {rec.id: rec.seq.__str__() for rec in SeqIO.parse(path, "fasta")}

def add_text_at_corner(myplt, text, where='top right', **kwargs):
    legal_pos = ['top right', 'top left', 'bottom right', 'bottom left']
    if where not in legal_pos:
        print ("where should be one of: " + ', '.join(legal_pos))
        return
    topbottom = where.split()[0]
    rightleft = where.split()[1]
    if str(type(myplt)) == "<class 'matplotlib.axes._subplots.AxesSubplot'>" or str(
            type(myplt)) == "<class 'mpl_toolkits.axes_grid1.parasite_axes.AxesHostAxes'>":
        x = myplt.get_xlim()
        y = myplt.get_ylim()
    elif str(type(myplt)) == "<type 'module'>":
        x = myplt.xlim()
        y = myplt.ylim()
    else:

        raise
    newaxis = {'left': x[0] + (x[1] - x[0]) * 0.01, 'right': x[1] - (x[1] - x[0]) * 0.01,
               'top': y[1] - (y[1] - y[0]) * 0.01, 'bottom': y[0] + (y[1] - y[0]) * 0.01}
    myplt.text(newaxis[rightleft], newaxis[topbottom], text, horizontalalignment=rightleft, verticalalignment=topbottom,
               **kwargs)


def PCA_transform(data, n_components=100, fillna='floor', take_log10=True, return_pca=False):
    print ('PCA_transform')
    assert type(data) is pd.core.frame.DataFrame
    data_copy = data.copy()
    if take_log10:
        if data_copy.min().min() >= 0 and data_copy.max().max() <= 1:
            data_copy[data_copy == 0] = np.nan
            data_copy = data_copy.apply(lambda x: np.log10(x))
    if isinstance(fillna, (int, np.long, float, complex)):
        data_copy.fillna(fillna, inplace=True)
    elif fillna is min:
        data_copy.fillna(data_copy.min().min(), inplace=True)
    elif fillna == 'floor':
        data_copy.fillna(np.floor(data_copy.min().min()), inplace=True)
    pca = PCA(n_components=n_components)
    #     pca.fit_transform(data_copy)
    a_scores = pca.fit_transform(data_copy.values)
    pca_df = pd.DataFrame(a_scores, index=data_copy.index,
                          columns=['PC' + str(i + 1) for i in range(a_scores.shape[1])])
    if return_pca:
        return pca_df, pca
    else:
        return pca_df

def make_dir_if_not_exists(path):
    if not os.path.exists(path): os.makedirs(path)


def pearsonr_rmna(x, y):
    try:
        X = x.values.ravel()
        Y = y.values.ravel()
    except:
        X = x.ravel()
        Y = y.ravel()
    mask = ~np.isnan(X) & ~np.isnan(Y)
    # mask array is now true where ith rows of df and dg are NOT nan.
    X = X[mask]  # this returns a 1D array of length mask.sum()
    Y = Y[mask]
    return pearsonr(X, Y)


def r2_score_rmna(x, y):
    try:
        X = x.values.ravel()
        Y = y.values.ravel()
    except:
        X = x.ravel()
        Y = y.ravel()
    mask = ~np.isnan(X) & ~np.isnan(Y)
    # mask array is now true where ith rows of df and dg are NOT nan.
    X = X[mask]  # this returns a 1D array of length mask.sum()
    Y = Y[mask]
    return r2_score(X, Y)


def spearmanr_minimal(x, y, return_values=(np.nan, np.nan), nan_policy='omit'):
    try:
        X = x.values.ravel()
        Y = y.values.ravel()
    except:
        X = x.ravel()
        Y = y.ravel()
    if len(set(X)) < 3 or len(set(Y)) < 3:
        return return_values
    else:
        return spearmanr(X, Y, nan_policy=nan_policy)


def remove_rare_elements(df, rare_def=0.05, null=True):
    """ Will keep only rows with at least rare_def percent non-null (or zero) values.
        Assumes rows are elements and columns are samples."""
    if null:
        rows2keep = df.notnull().sum(1) > float(df.shape[1] * rare_def)
    else:
        df_min = df.min().min()
        rows2keep = (df > df_min).sum(1) > float(df.shape[1] * rare_def)
    print ('removing %0.3f of elements.' % (1 - float(rows2keep.sum()) / len(rows2keep)))
    return df.loc[rows2keep].copy()


def prepare_df_for_pca(df):
    print ('prepare_df_for_pca')
    df_min = df.min().min()
    if df_min >= 0:
        df = df.fillna(0)
    else:
        df.fillna(np.floor(df_min), inplace=True)
    return df


def change_index(data, df, from_idx, to_idx):
    c_name = data.columns.names
    data = data.reset_index().merge(df.reset_index()[[from_idx, to_idx]], on=from_idx).set_index(to_idx).copy()
    del data[from_idx]
    data = data.loc[data.index.notnull()]
    data.columns.names = c_name
    return data


def robust_zs(df):
    df_tmp = df.clip(lower=df.quantile(.05), upper=df.quantile(.95), axis=1)
    return ((df - df.median()) / df_tmp.std())


def robust_clipping(df, n_stds=5, median_based=False, lower_upper=(0.05, 0.95)):
    """

    :param df:
    :param n_stds:
    :param median_based:
    :param lower_upper:
    :return:
    """
    df_tmp = df.clip(lower=df.quantile(lower_upper[0]), upper=df.quantile(lower_upper[1]), axis=1)
    if median_based:
        return df.clip(lower=df_tmp.median() - n_stds*df_tmp.std(), upper=df_tmp.median() + n_stds*df_tmp.std(), axis=1)
    return df.clip(lower=df_tmp.mean() - n_stds*df_tmp.std(), upper=df_tmp.mean() + n_stds*df_tmp.std(), axis=1)



def trim_values(df, quantiles=(0.05, 0.95)):
    return df[(df > df.quantile(quantiles[0])) & (df < df.quantile(quantiles[1]))].copy()


def flatten_list(l):
    return [item for sublist in l for item in sublist]


###################################################################################################
# Metabolon general functions
###################################################################################################

def clip_outliers(df, n_stds=4, clip_or_na='clip'):
    if clip_or_na not in ['clip', 'na']:
        print ('clip_or_na must be one of ' + ', '.join(['clip', 'na']))
        return None
    for c in tqdm_notebook(df.columns, desc='clip_outliers'):
        outlier_th = n_stds * df[c].std()
        df_mean = df[c].mean()
        if df[c].unique().shape[0] < 5: print (str(c) + ' is probably a categorical variable, not clipping.'); continue
        if clip_or_na == 'na':
            is_outlier = np.abs(df[c] - df_mean) > outlier_th
            if (np.sum(is_outlier) > 0): df.loc[is_outlier, c] = np.nan
        if clip_or_na == 'clip':
            is_upper_outlier = df[c] - df_mean > outlier_th
            if (np.sum(is_upper_outlier) > 0): df.loc[is_upper_outlier, c] = df_mean + outlier_th
            is_lower_outlier = df_mean - df[c] > outlier_th
            if (np.sum(is_lower_outlier) > 0): df.loc[is_lower_outlier, c] = df_mean - outlier_th
    return df


###################################################################################################
# Computing absolute and sign shap values
###################################################################################################

def compute_abs_SHAP(dic):
    abs_shap = pd.DataFrame(index=dic.keys(), columns=dic[dic.keys()[0]].columns)
    for k in tqdm_notebook(dic, desc='compute_abs_SHAP'):
        abs_shap.loc[k, :] = dic[k].apply(np.abs).apply(np.mean).values.ravel()
    return abs_shap


def compute_signed_abs_SHAP(dic, X):
    signed_shap = pd.DataFrame(index=dic.keys(), columns=dic[dic.keys()[0]].columns)
    for k in tqdm_notebook(dic.keys(), desc='compute_signed_abs_SHAP'):
        temp_X = X.loc[dic[k].index].copy()
        temp_dic = dic[k].copy()
        #         for c in signed_shap.columns:
        #             signed_shap.loc[k, c] = np.sign(spearmanr(temp_X[c], temp_dic[c], nan_policy='omit')[0])
        signed_shap.loc[k, :] = temp_X.apply(lambda x: np.sign(spearmanr_minimal(x, temp_dic[x.name], nan_policy='omit')[0]))
    return signed_shap


def log_run_details(command_args):
    """

    :param command_args:
    :return:
    """
    with open(command_args.output_dir + '/log_run_' + str(datetime.now()).split('.')[0].replace(' ', '_') + '.txt',
              'w') as handle:
        handle.write('### Arguments ###\n')
        handle.write(str(sys.argv[0]) + '\n')
        for arg in vars(command_args):
            handle.write(str(arg) + '\t' + str(getattr(command_args, arg)) + '\n')
        handle.write('\n### Code ###\n')
        with open(sys.argv[0], 'r') as f:
            for l in f.readlines():
                handle.write(l)


def directed_mannwhitneyu(x, y, use_continuity=True):
    """
    Copy of scipy.stats.mannwhitneyu which multiplies the static by the direction.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n1 = len(x)
    n2 = len(y)
    ranked = rankdata(np.concatenate((x, y)))
    rankx = ranked[0:n1]  # get the x-ranks
    u1 = n1 * n2 + (n1 * (n1 + 1)) / 2.0 - np.sum(rankx, axis=0)  # calc U for x
    u2 = n1 * n2 - u1  # remainder is U for y
    bigu = max(u1, u2)
    smallu = min(u1, u2)
    T = tiecorrect(ranked)
    if T == 0:
        raise ValueError('All numbers are identical in amannwhitneyu')
    sd = np.sqrt(T * n1 * n2 * (n1 + n2 + 1) / 12.0)

    if use_continuity:
        # normal approximation for prob calc with continuity correction
        z = abs((bigu - 0.5 - n1 * n2 / 2.0) / sd)
    else:
        z = abs((bigu - n1 * n2 / 2.0) / sd)  # normal approximation for prob calc
    return (eps if smallu == 0 else smallu) * (-1 if smallu == u1 else 1), distributions.norm.sf(
        z)  # (1.0 - zprob(z))

def p_base10(p):
    return np.ceil(np.log10(p))

def add_null_indicator_column(df, null=np.nan, fillna=None, prefix='indicator'):
    """
    adding indicator columns to dataframe based on some value
    :param df:
    :param null:
    :param prefix:
    :return:
    """
    new_df = df.copy()
    for col in df.columns:
        if null is np.nan:
            if df[col].isnull().sum() > 0:
                new_df[prefix + '_' + col] = df[col].isnull().astype(int).copy()
                if fillna is not None:
                    new_df[col].fillna(fillna, inplace=True)
        else:
            if (df[col] == null).sum() > 0:
                new_df[prefix + '_' + col] = (df[col] == null).astype(int).copy()
                if fillna is not None:
                    new_df[col].replace(null, fillna, inplace=True)
    return new_df

def get_nlargest_cols(df, n=3, col_name='column', val_name='value', index_name='CHEMICAL_ID'):
    """

    :param df:
    :param n:
    :param index_name:
    :return:
    """
    df_names = df.apply(lambda s: s.abs().nlargest(n).index.tolist(), axis=1).apply(pd.Series).copy()
    df_names.columns = ['%s #%0.1d'%(col_name, d) for d in range(1,n+1)]
    df_names.index.names = [index_name]
    df_values = df.apply(lambda s: df.loc[s.name, s.abs().nlargest(n).index.tolist()].values.tolist(), axis=1).apply(pd.Series).copy()
    df_values.columns = ['%s #%0.1d'%(val_name, d) for d in range(1,n+1)]
    df_values.index.names = [index_name]
    df_top = pd.concat((df_names, df_values), axis=1).rename_axis(index_name, axis=0)
    df_top = df_top.iloc[:, flatten_list([[i, i+n] for i in range(n)])]
    return df_top


def balance_sampleing(df):
    """
    takes a data frame with a single column representing the class, and over samples the indexes such that all classes
    will be of same size (equal to the largest class).
    Keeps all original samples, and randomly samples from the smaller classes.
    :param df:
    :return:
    """
    # get the number needed for sampling as max label minus number of samples per label
    col_name = df.columns[0]
    n_samples_per_label = pd.Series(df.loc[:, col_name]).value_counts()
    max_samples = n_samples_per_label.max()
    n_to_sample = max_samples - n_samples_per_label
    return pd.concat([df] + [df.loc[np.random.choice(df[df[col_name] == label].index, n_to_sample.loc[label], replace=True), :] for
                             label in n_to_sample.index], axis=0, sort=False)


########################################################################################################################
# Nightingale
########################################################################################################################

def calculate_LDL_mg_to_dL(df):
    """
    compute the LDL cholesterol in mg/dL
    :param df: A nightingale data frame
    :return:
    """
    return (df['LDL_C'] + df['IDL_C'] + df['XS_VLDL']*0.15)*38.67

