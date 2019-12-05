import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import os
import sys
import pickle
import json
import pandas as pd
import seaborn as sns
from random import shuffle
from itertools import chain
from sklearn.metrics import r2_score

def mk_subjwise_ts_dict(clip, gsr=0, start_stop_pads = (10, 5), total_trs = None, subj_list='subj_list.npy', data_dir='/data/HCP_preproc/7T_movie/cpm/data/all_shen268_roi_ts/'):

    subj_list = np.load(subj_list)
    video_tr_lookup = pd.read_csv('/data/HCP_preproc/7T_movie/cpm/data/video_tr_lookup.csv')
    subjwise_ts_dict = {}

    run_name_dict = {
                    "MOVIE1": "MOVIE1_7T_AP",
                    "MOVIE2": "MOVIE2_7T_PA",
                    "MOVIE3": "MOVIE3_7T_PA",
                    "MOVIE4": "MOVIE4_7T_AP"
                    }

    if clip in ["MOVIE1", "MOVIE2", "MOVIE3", "MOVIE4"]:
        run_name = run_name_dict[clip]
        start_tr = 0
        stop_tr = None
    else:
        # figure out which run this clip is in, get start and stop trs
        run_name = video_tr_lookup.query('clip_name==@clip')["run"].tolist()[0]
        start_tr = video_tr_lookup.loc[video_tr_lookup["clip_name"]==clip, "start_tr"].values[0]
        stop_tr = video_tr_lookup.loc[video_tr_lookup["clip_name"]==clip, "stop_tr"].values[0]
        start_tr+= start_stop_pads[0]
        stop_tr+=start_stop_pads[1]

    if gsr ==1:
        f_suffix = "_" + run_name + "_shen268_roi_ts_gsr.txt"
    elif gsr ==0:
        f_suffix = "_" + run_name + "_shen268_roi_ts.txt"

    for s,subj in enumerate(subj_list):
        f_name = data_dir + subj + f_suffix
        tmp_run = pd.read_csv(f_name, sep='\t', header=None).dropna(axis=1)
        if s==0:
            print(tmp_run.shape)
        tmp_run = tmp_run.iloc[start_tr:stop_tr, :] # take only desired TRs
        if s==0:
            print(tmp_run.shape)
        subjwise_ts_dict[subj] = sp.stats.zscore(tmp_run)

    return subjwise_ts_dict

# ----------------------------------------------------------------------------------------------------
def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))

# ----------------------------------------------------------------------------------------------------
def get_fc_vcts(subjwise_ts_dict, zscore = False):
    """
    Extracts per-subject timeseries for a given clip (or whole MOVIE run)
    from subjwise_ts_dict and creates individual FC matrices
    Returns dataframe that is subjects x edges
    """


    # Initialize result
    subj_list = list(subjwise_ts_dict)
    n_subs = len(subj_list)
    n_nodes = subjwise_ts_dict[subj_list[0]].shape[1] # get a random subject's timeseries to check number of nodes
    n_edges = int(n_nodes*(n_nodes-1)/2)
    df = pd.DataFrame(np.zeros((n_subs,n_edges)), index=subj_list)

    # Get triu indices
    iu1 = np.triu_indices(n_nodes, k=1)

    # Get timeseries for each subj
    for s,subj in enumerate(subj_list):
        this_subj_ts = subjwise_ts_dict[subj]
        # this_subj_vct = np.corrcoef(this_subj_ts.T)[iu1]
        this_subj_vct = corr2_coeff(this_subj_ts.T, this_subj_ts.T)[iu1] # this way is a bit faster

        if zscore==1:
            this_subj_vct = sp.stats.zscore(this_subj_vct)
        df.loc[subj,:] = this_subj_vct

    return df

# ----------------------------------------------------------------------------------------------------
def mk_kfold_indices(family_list='family_list.npy', k = 10):
    """
    Splits list of subjects into k folds, respecting family structure.
    """
    family_list = np.load(family_list)
    n_fams = len(family_list)
    n_fams_per_fold = n_fams//k # floor integer for n_fams_per_fold

    indices = [[fold_no]*n_fams_per_fold for fold_no in range(k)] # generate repmat list of indices
    remainder = n_fams % k # figure out how many subs are leftover
    remainder_inds = list(range(remainder))
    indices = list(chain(*indices)) # flatten list
    [indices.append(ind) for ind in remainder_inds] # add indices for remainder subs

    assert len(indices)==n_fams, "Length of indices list does not equal number of families, something went wrong"

    shuffle(indices) # shuffles in place

    return np.array(indices)

# ----------------------------------------------------------------------------------------------------
def get_train_test_subs_for_kfold(indices, test_fold, family_list='family_list.npy'):
    """
    For a given fold, family list, and k-fold indices, returns lists of train_subs and test_subs
    """
    family_list = np.load(family_list)

    train_inds = np.where(indices!=test_fold)
    test_inds = np.where(indices==test_fold)

    # Flatten lists
    train_subs = []
    for sublist in family_list[train_inds]:
        for item in sublist:
            train_subs.append(item)

    test_subs = []
    for sublist in family_list[test_inds]:
        for item in sublist:
            test_subs.append(item)

    return (train_subs, test_subs)

# ----------------------------------------------------------------------------------------------------
def get_train_test_data(all_fc_data, train_subs, test_subs, behav_data, behav):

    """
    Extracts requested FC and behavioral data for a list of train_subs and test_subs
    """

    train_vcts = all_fc_data.loc[train_subs, :]
    test_vcts = all_fc_data.loc[test_subs, :]

    train_behav = behav_data.loc[train_subs, behav]
    # test_behav = behav_data.loc[test_subs, behav]

    return (train_vcts, train_behav, test_vcts)

# ----------------------------------------------------------------------------------------------------
def select_features(train_vcts, train_behav, r_thresh, corr_type='pearson'):

    assert train_vcts.index.equals(train_behav.index), "Row indices of FC vcts and behavior don't match!"

    # Correlate all edges with behav vector
    if corr_type =='pearson':
        cov = np.dot(train_behav.T - train_behav.mean(), train_vcts - train_vcts.mean(axis=0)) / (train_behav.shape[0]-1)
        corr = cov / np.sqrt(np.var(train_behav, ddof=1) * np.var(train_vcts, axis=0, ddof=1))
    elif corr_type =='spearman':
        corr = []
        for edge in train_vcts.columns:
            r_val = sp.stats.spearmanr(train_Vcts.loc[:,edge], train_behav)[0]
            corr.append(r_val)

    # Define positive and negative masks
    mask_dict = {}
    mask_dict["pos"] = corr > r_thresh
    mask_dict["neg"] = corr < -r_thresh

    return mask_dict

# ----------------------------------------------------------------------------------------------------
def build_model(train_vcts, mask_dict, train_behav):

    assert train_vcts.index.equals(train_behav.index), "Row indices of FC vcts and behavior don't match!"

    model_dict = {}

    # Loop through pos and neg tails
    X_glm = np.zeros((train_vcts.shape[0], len(mask_dict.items())))

    t = 0
    for tail, mask in mask_dict.items():
        X = train_vcts.values[:, mask].sum(axis=1)
        X_glm[:, t] = X
        y = train_behav
        (slope, intercept) = np.polyfit(X, y, 1)
        model_dict[tail] = (slope, intercept)
        t+=1

    X_glm = np.c_[X_glm, np.ones(X_glm.shape[0])]
    model_dict["glm"] = tuple(np.linalg.lstsq(X_glm, y, rcond=None)[0])

    return model_dict

# ----------------------------------------------------------------------------------------------------
def apply_model(test_vcts, mask_dict, model_dict):

    behav_pred = {}

    X_glm = np.zeros((test_vcts.shape[0], len(mask_dict.items())))

    # Loop through pos and neg tails
    t = 0
    for tail, mask in mask_dict.items():
        X = test_vcts.loc[:, mask].sum(axis=1)
        X_glm[:, t] = X

        slope, intercept = model_dict[tail]
        behav_pred[tail] = slope*X + intercept
        t+=1

    X_glm = np.c_[X_glm, np.ones(X_glm.shape[0])]
    behav_pred["glm"] = np.dot(X_glm, model_dict["glm"])

    return behav_pred

# ----------------------------------------------------------------------------------------------------
def evaluate_predictions(behav_pred, behav_obs):

    accuracies = pd.DataFrame(index=behav_pred.columns, columns=["pearson", "spearman", "r2"])

    for tail in list(behav_pred.columns):

        x = behav_obs
        y = behav_pred[tail]

        accuracies.loc[tail, "pearson"] = sp.stats.pearsonr(x,y)[0]
        accuracies.loc[tail, "spearman"] = sp.stats.spearmanr(x,y)[0]
        accuracies.loc[tail, "r2"] = r2_score(x,y)

    return accuracies

# ----------------------------------------------------------------------------------------------------
