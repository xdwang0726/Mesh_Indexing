#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:08:58 2019

@author: wangxindi
"""

# anaylsis predicted meshs are more precise or more general 
import numpy as np
from build_graph import Graph

connections = []
with open('MeSH_parent_child_mapping_2018.txt') as f:
    for line in f:
        item = tuple(line.strip().split(" "))
        connections.append(item)
        
# build a graph
g_undirected = Graph(connections, directed=False)
g_directed = Graph(connections, directed=True)

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))

def find_chirdren(y, distance):    
    new_y = []
    for item in y:
        y_child = g_directed.find_node(item, distance)
        #y_new = list(set(y_child+item))
        new_y.append(y_child)
    #new_y = [x for x in new_y if x != []]
    #new_y = [item for sublist in new_y for item in sublist]
    return new_y

def find_parent(y, distance):   
    new_y = []
    for i in range(len(y)):
        y_child = g_directed.find_node(y[i], distance)
        y_parent_child = g_undirected.find_node(y[i], distance)
        y_parent = list(set(y_parent_child) - set(y_child))
        #y_new = list(set(y_parent+y[i]))
        y_new = list(set(y_parent) - set(y[i]))
        new_y.append(y_new)
    return new_y

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

def hierachy_eval_parent(y_actural, y_hat, distance):
    
    #new_y_actural = find_parent(y_actural, distance)
    new_y_hat = find_parent(y_hat, distance)
    num_common_label = find_common_label(y_actural, new_y_hat)
    
    avg = np.mean(num_common_label)
    total = np.sum(num_common_label)
    result = [avg, total]
    return result   

def hierachy_eval_child(y_actural, y_hat, distance):
    
    #new_y_actural = find_chirdren(y_actural, distance)
    new_y_hat = find_chirdren(y_hat, distance)
    num_common_label = find_common_label(y_actural, new_y_hat)
    
    #num  = []
    #for item in num_common_label:
    #    length = len(item)
    #    num.append(length)
    
    avg = np.mean(num_common_label)
    total = np.sum(num_common_label)
    result = [avg, total]
    return result

def element_wise_merge_list(list1, list2):
    new_list = []
    if len(list1) != len(list2):
        print("Error!")
    else:
        for i in range(len(list1)):
            newitem = list(set(list1[i]+list2[i]))
            new_list.append(newitem)
    return new_list
            

def hierachy_analysis(y_actural, y_hat, distance_children, distance_parents):
    #y_actural_child = find_chirdren(y_actural, distance_children)
    #y_actural_parents = find_parent(y_actural, distance_parents)
    #new_y_actural = element_wise_merge_list(y_actural_child, y_actural_parents)
    
    y_hat_child = find_chirdren(y_hat, distance_children)
    y_hat_parents = find_parent(y_hat, distance_parents)
    new_y_hat = element_wise_merge_list(y_hat_child, y_hat_parents)
    
    num_common_label = find_common_label(y_actural, new_y_hat)
    
    avg = np.mean(num_common_label)
    total = np.sum(num_common_label)
    result = [avg, total]
    return result
        

with open("TextCNN_true_label.txt", "r") as true:
    true_mesh = true.readlines()
goldenStandrad = []
for mesh in true_mesh:
    mesh = mesh.split()
    goldenStandrad.append(mesh)
    
with open("TextCNN_pred_label_5.txt", "r") as pred:
    pred_mesh = pred.readlines()
predicted = []
for mesh in pred_mesh:
    mesh = mesh.split()
    predicted.append(mesh)

print("TextCNN top5 predicted:")

# distance 1
hierachy_upper1 = hierachy_eval_parent(goldenStandrad, predicted, 1)
print("HP@1_c0_p1:", hierachy_upper1)
    
hierachy_lower1 = hierachy_eval_child(goldenStandrad, predicted, 1)
print("HP@1_c1_p0:", hierachy_lower1)

# distance 2
hierachy_upper2 = hierachy_eval_parent(goldenStandrad, predicted, 2)  
print("HP@2_c0_p2:", hierachy_upper2)

hierachy_lower2 = hierachy_eval_child(goldenStandrad, predicted, 2)
print("HP@2_c2_p0:",hierachy_lower2)


hierachy_p1_c2 = hierachy_analysis(goldenStandrad, predicted, 2, 1)
print("HP@2_c2_p1:", hierachy_p1_c2)
  
hierachy_p2_c1 = hierachy_analysis(goldenStandrad, predicted, 1, 2)
print("HP@2_c1_p2:", hierachy_p2_c1)

