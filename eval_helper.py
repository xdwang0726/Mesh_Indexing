#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 02:00:00 2018

@author: wangxindi
"""

import numpy as np

from scipy.sparse import issparse
from scipy import stats

from build_graph import Graph

connections = []
with open('MeSH_parent_child_mapping_2018.txt') as f:
    for line in f:
        item = tuple(line.strip().split(" "))
        connections.append(item)
        
# build a graph
g = Graph(connections, directed=False)

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))

def precision(p, t):
    """
    p, t: two sets of labels/integers
    >>> precision({1, 2, 3, 4}, {1})
    0.25
    """
    return len(t.intersection(p)) / len(p)


def precision_at_ks(Y_pred_scores, Y_test, ks):
    """
    Y_pred_scores: nd.array of dtype float, entry ij is the score of label j for instance i
    Y_test: list of label ids
    """
    result = []
    for k in ks:
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
    result = actual / best
    return round(result,5)

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
    TP_total = []
    FP_total = []
    TN_total = []
    FN_total = []

    for i in range(y_actual.shape[1]): 
        TP = 1
        FP = 1
        TN = 1
        FN = 1
        
        for j in range(y_actual.shape[0]):
            if y_actual[j,i]==y_hat[j,i]==1:
                TP += 1
            if y_hat[j,i]==1 and y_actual[j,i]!=y_hat[j,i]:
                FP += 1
            if y_actual[j,i]==y_hat[j,i]==0:
                TN += 1
            if y_hat[j,i]==0 and y_actual[j,i]!=y_hat[j,i]:
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
    
    result = [round(MaP, 5), round(MiP,5), round(MaF,5), round(MiF,5)]
    return result

def example_based_precision(CL,y_hat):
    
    EBP = []
    
    for i in range(len(CL)):
        ebp = CL[i]/len(y_hat[i])
        EBP.append(ebp)
        
    EBP = np.mean(EBP)
    return EBP

def example_based_recall(CL, y_actural):
    
    EBR = []
    
    for i in range(len(CL)):
        ebr = CL[i]/len(y_actural[i])
        EBR.append(ebr)
    EBR = np.mean(EBR)
    return EBR

def example_based_fscore(CL, y_actual,y_hat):
    
    EBF = []
    
    for i in range(len(CL)):
        ebf = (2*CL[i])/(len(y_hat[i]) + len(y_actual[i]))
        EBF.append(ebf)
    
    EBF = np.mean(EBF)
    return EBF

def find_common_label(y_actual, y_hat):
    
    num_common_label = []
    
    for i in range(len(y_actual)):
        labels = intersection(y_actual[i],y_hat[i])
        num_label = len(labels)
        num_common_label.append(num_label)
    return num_common_label
    

def example_based_evaluation(y_actual, y_hat):
    
    num_common_label = find_common_label(y_actual, y_hat)
  
    EBP = example_based_precision(num_common_label, y_hat)
    EBR = example_based_recall(num_common_label, y_actual)
    EBF = example_based_fscore(num_common_label, y_actual, y_hat)
    result = [round(EBP,5), round(EBR,5), round(EBF,5)]
    return result


#def hierachy_EBP(num_original_CL, num_distance_CL,len_dis_CL, y_hat):
#    EBP = []          
#    for i in range(len(num_original_CL)):
#        temp_value = num_original_CL[i]
#        for j in range(len(num_distance_CL)):
#            value = num_distance_CL[j][i]
#            temp_value = temp_value + value
#            temp_len = len(y_hat[i]) + len_dis_CL[j][i]
#        ebp = temp_value/temp_len
#        EBP.append(ebp)
#    EBP = np.mean(EBP)
#    return EBP

#def hierachy_EBR(num_original_CL, num_distance_CL, y_actural):
#    EBR = []          
#    for i in range(len(num_original_CL)):
#        temp = num_original_CL[i]
#        for j in range(len(num_distance_CL)):
#            value = num_distance_CL[j][i]
#            temp = temp + value
#        ebr = temp/len(y_actural[i])
#        EBR.append(ebr)
#    EBR = np.mean(EBR)
#    return EBR

#def hierachy_fscore(num_original_CL, num_distance_CL, len_dis_CL, y_actural, y_hat):
#    EBF = []          
#    for i in range(len(num_original_CL)):
#        temp = num_original_CL[i]
#        for j in range(len(num_distance_CL)):
#            value = num_distance_CL[j][i]
#            temp = temp + value
#            temp_len = len(y_hat[i]) + len_dis_CL[j][i]
#        ebf = 2*temp/(temp_len + len(y_actural[i]))  
#        EBF.append(ebf)
#    EBF = np.mean(EBF)
#    return EBF


def find_node_set(y, distance):
    
    new_y = []
    for i in range(len(y)):
        y_parents = g.find_node(y[i], distance)
        y_new = list(set(y_parents+y[i]))
        new_y.append(y_new)
    return new_y


#def hierachy(y_actural, y_hat, distance):
    
#    original_CL = find_common_label(y_actural, y_hat)
    
#    num_CL_with_distance = []
#    len_num_CL_distance = []
#    for i in range(distance):
#        new_y_hat = find_node_set(y_hat, i+1)
#        len_new_y_hat = [len(item) for item in new_y_hat]
#        len_num_CL_distance.append(len_new_y_hat)
#        num_CL = find_common_label(y_actural, new_y_hat)
#        num_CL_with_distance.append(num_CL)


#    num_difference = []
#    for j in range(len(num_CL_with_distance)-1):
#       value = [b - a for a, b in zip(num_CL_with_distance[j], num_CL_with_distance[j+1])]
#       num_difference.append(value)
    
#    num_difference.insert(1, num_CL_with_distance[0])
    
#    EBP = hierachy_EBP(original_CL, num_difference, len_num_CL_distance, y_hat)
#    EBR = hierachy_EBR(original_CL, num_difference, y_actural)
#    EBF = hierachy_fscore(original_CL, num_difference, len_num_CL_distance, y_actural, y_hat)
#    result = [round(EBP,5), round(EBR,5), round(EBF,5)]
#    return result


def hierachy_eval(y_actural, y_hat, distance):
    
    new_y_actural = find_node_set(y_actural, distance)
    new_y_hat = find_node_set(y_hat, distance)
    num_common_label = find_common_label(new_y_actural, new_y_hat)
    
    HP = example_based_precision(num_common_label, new_y_hat)
    HR = example_based_recall(num_common_label, new_y_actural)
    result = [round(HP,5), round(HR,5)]
    return result       
  
