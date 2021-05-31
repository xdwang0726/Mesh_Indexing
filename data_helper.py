#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 14:29:03 2018

@author: wangxindi

This file is used to preprocess text 

1. precentage XX% convert to "PERCENTAGE"
2. Chemical numbers(word contains both number and letters) to "Chem"
3. All numbers convert to "NUM"
4. Mathematical symbol （=, <, >, >/=, </= ）
5. "-" replace with "_"
6. remove punctuation
7. covert to lowercase

"""

import re
import numpy as np
from keras.preprocessing import sequence


def text_preprocess(string):
    string = re.sub("\\d+(\\.\\d+)?%", "Percentage", string) 
    string = re.sub("((?:[a-zA-Z]+[0-9]|[0-9]+[a-zA-Z])[a-zA-Z0-9]*)", "Chemical", string)
    string = re.sub(r'[0-9]+', 'Num', string)
    string = re.sub("=", "Equal", string)
    string = re.sub(">", "Greater", string)
    string = re.sub("<", "Less", string)
    string = re.sub(">/=", "GreaterAndEqual", string)
    string = re.sub("</=", "LessAndEqual", string)
    string = re.sub("-", "_", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub("[.,?;*!%^&+():\[\]{}]", " ", string)
    string = string.replace('"', '')
    string = string.replace('/', '')
    string = string.replace('\\', '')
    string = string.replace("'", '')
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def padding(input_data,maxlen):
        padded_data = sequence.pad_sequences(input_data, maxlen)
        return padded_data

def batch_iter(input_x, input_y, input_y_index, batch_size, num_epochs, max_seq_len, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    input_x = np.array(input_x)
    data_size = len(input_x)
    num_batches_per_epoch = int((len(input_x)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_x = input_x[shuffle_indices]
            shuffled_y = input_y[shuffle_indices]
            shuffled_y_index = input_y_index[shuffle_indices]
        else:
            shuffled_x = input_x
            shuffled_y = input_y
            shuffled_y_index = input_y_index
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            shuffled_x = shuffled_x[start_index:end_index]
            padded_shuffled_x = padding(shuffled_x, max_seq_len)
            shuffled_y = shuffled_y[start_index:end_index]
            shuffled_y_index = shuffled_y_index[start_index:end_index]
            yield padded_shuffled_x, shuffled_y, shuffled_y_index