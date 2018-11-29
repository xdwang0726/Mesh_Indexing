#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 02:00:00 2018

@author: wangxindi
"""

import numpy as np

from scipy.sparse import issparse


def precision(p, t):
    """
    p, t: two sets of labels/integers
    >>> precision({1, 2, 3, 4}, {1})
    0.25
    """
    return len(t.intersection(p)) / len(p)


def precision_at_ks(Y_pred_scores, Y_test, ks=[1, 3, 5, 10]):
    """
    Y_pred_scores: nd.array of dtype float, entry ij is the score of label j for instance i
    Y_test: list of label ids
    """
    result = []
    for k in [1, 3, 5, 10]:
        Y_pred = []
        for i in np.arange(Y_pred_scores.shape[0]):
            if issparse(Y_pred_scores):
                idx = np.argsort(Y_pred_scores[i].data)[::-1]
                Y_pred.append(set(Y_pred_scores[i].indices[idx[:k]]))
            else:  # is ndarray
                idx = np.argsort(Y_pred_scores[i, :])[::-1]
                Y_pred.append(set(idx[:k]))

        result.append(np.mean([precision(yp, set(yt)) for yt, yp in zip(Y_test, Y_pred)]))
    return result

def dcg_score(y_true, y_score, k, gains="linear"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k, gains="linear"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best

def macro_precision(TP, FP):
    MaP = []
    for i in range(len(TP)):
        macro_p = TP[i]/(TP[i] + FP[i])
        MaP.append(macro_p)
    MaP = np.mean(MaP)
    return MaP

def macro_recall(TP, FN):
    MaR = []
    for i in range(len(TP)):
        macro_r = TP[i]/(TP[i] + FN[i])
        MaR.append(macro_r)
    MaR = np.mean(MaR)
    return MaR

def micro_precision(TP, FP):
    MiP = sum(TP)/(sum(TP) + sum(FP))
    return MiP

def micro_recall(TP, FN):
    MiR = sum(TP)/(sum(TP) + sum(FN))
    return MiR

def macro_f1(MaP, MaR):
    MaF = stats.hmean([MaP, MaR])
    return MaF

def micro_f1(MiP, MiR):
    MiF = stats.hmean([MiP, MiR])
    return MiF

def perf_measure(y_actual, y_hat):
    result = []
    TP_total = []
    FP_total = []
    TN_total = []
    FN_total = []

    for i in range(y_actual.shape[0]): 
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for j in range(y_actual.shape[1]):
            if y_actual[i,j]==y_hat[i,j]==1:
                TP += 1
            if y_hat[i,j]==1 and y_actual[i,j]!=y_hat[i,j]:
                FP += 1
            if y_actual[i,j]==y_hat[i,j]==0:
                TN += 1
            if y_hat[i,j]==0 and y_actual[i,j]!=y_hat[i,j]:
                FN += 1         
        TP_total.append(TP)
        FP_total.append(FP)
        TN_total.append(TN)
        FN_total.append(FN)
    
    MaP = macro_precision(TP_total, FP_total)
    MiP = micro_precision(TP_total, FP_total)
    MaR = macro_recall(TP_total, FN_total)
    MiR = micro_recall(TP_total, FN_total)
    MaF = macro_f1(MaP, MaR)
    MiF = micro_f1(MiP, MiR)
    
    result = [MaP, MiP, MaF, MiF]
    return result
