#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 15:58:25 2018

@author: wangxindi
"""

"""
This script is used to calculate the basic statistics of the dataset

** calculate std, median later **

"""

import numpy as np
from data_helper import text_preprocess

import statistics
########## Data preprocess ##########
with open("AbstractAndTitle.txt", "r") as abAndTitle_token:
    abAndTitletoken = abAndTitle_token.readlines()
    
ab_title = []
ab_length = len(abAndTitletoken)    
for i in range(0, ab_length):
    token = abAndTitletoken[i].lstrip('0123456789.- ')
    token = text_preprocess(token)
    ab_title.append(token)

doc_dim = len(ab_title)

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(ab_title)

x_seq = tokenizer.texts_to_sequences(ab_title)
doc_len = [len(seq) for seq in x_seq]
avg_doc_length = sum(doc_len) / doc_dim
print("Average of words per document:", avg_doc_length) 
    
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(ab_title)

x_seq = tokenizer.texts_to_sequences(ab_title)
word_index = tokenizer.word_index

print('Total number of features: %s .' % len(word_index))



from sklearn.preprocessing import MultiLabelBinarizer
# read full mesh list 
with open("MeshIDListSmall.txt", "r") as ml:
    meshList = ml.readlines()

mesh_out = []
for mesh in meshList:
    mesh_term = mesh.lstrip('0123456789| .-')
    mesh_term = mesh_term.split("|")
    mesh_term = [ids.strip() for ids in mesh_term]
    mesh_out.append(mesh_term)

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(mesh_out)
label_dim = labels.shape[1]
print('Total number of class labels:', label_dim)


# total number of labels per doc
label_row_total = np.sum(labels, axis = 1)
label_bar = np.sum(label_row_total) / doc_dim
# standard deviation 
label_std = np.std(label_row_total)
# median 
label_median = statistics.median(label_row_total)
print('Total number of labels per document: ', label_bar)
print('median: ', label_median)
print('standard deviation: ', label_std)

# total number of document per label
label_col_total = np.sum(labels, axis = 0)
label_tilde = np.sum(label_col_total) / label_dim
print('Total number of documents per label: ', label_tilde)

