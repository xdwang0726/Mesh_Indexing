#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 17:05:09 2018

@author: wangxindi
"""

import time
import os
import numpy as np
from data_helper import text_preprocess
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence
from sklearn.preprocessing import MultiLabelBinarizer
import random

from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.models import Model

import pickle


from sklearn.metrics import hamming_loss
from eval_helper import precision_at_ks, ndcg_score, perf_measure

start = time. time()
#### GPU specified ####
os.environ["CUDA_DEVUCE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

########## FIXED PARAMETERS ##########
BATCH_SIZE = 10
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2

########## Data preprocess ##########
with open("AbstractAndTitle.txt", "r") as data_token:
    datatoken = data_token.readlines()

datatoken = list(filter(None, datatoken))
print('Total number of document: %s' % len(datatoken))
    
load_data = []
data_length = len(datatoken)    
for i in range(0, data_length):
    token = datatoken[i].lstrip('0123456789.- ')
    token = text_preprocess(token)
    load_data.append(token)
    
tokenizer = Tokenizer()
tokenizer.fit_on_texts(load_data)

x_seq = tokenizer.texts_to_sequences(load_data)
print("x_seq: %s" % len(x_seq))
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

def second_largest(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1            
            else:
                m2 = x
    return m2 if count >= 2 else None

# find the maximum length of document
MAX_SEQUENCE_LENGTH = max([len(seq) for seq in x_seq])
seconde_largest_length= second_largest([len(seq) for seq in x_seq])
print("Max sequence length: %s " % MAX_SEQUENCE_LENGTH)
print("Second largest sequence length: %s" % seconde_largest_length)

if MAX_SEQUENCE_LENGTH >= seconde_largest_length:
    MAX_SEQUENCE_LENGTH = seconde_largest_length
print("Padding size: %s" % MAX_SEQUENCE_LENGTH)

# read full meshIDs
with open("MeshIDList.txt", "r") as ml:
    meshIDs = ml.readlines()
    
meshIDs = [ids.strip() for ids in meshIDs]
label_dim = len(meshIDs)
mlb = MultiLabelBinarizer(classes = meshIDs)
print("Lable dimension: ", label_dim)

# read full mesh list 
with open("MeshIDListSmall.txt", "r") as ml:
    meshList = ml.readlines()

mesh_out = []
for mesh in meshList:
    mesh_term = mesh.lstrip('0123456789| .-')
    mesh_term = mesh_term.split("|")
    mesh_term = [ids.strip() for ids in mesh_term]
    mesh_out.append(mesh_term)

    
VALIDATION_SPLIT = 0.2

# shuffle data
data = []
mesh_label = []
indices = np.arange(len(x_seq))
np.random.shuffle(indices)
print("Indices length: %s" % len(indices))
for i in indices:
    data.append(x_seq[i])
    mesh_label.append(mesh_out[i])


def getLabelIndex(labels):
    label_index = np.zeros((len(labels), len(labels[1])))
    for i in range(0, len(labels)):
        index = np.where(labels[i] == 1)
        index = np.asarray(index)
        N = len(labels[1])-index.size
        index = np.pad(index, [(0, 0),(0, N)], 'constant')
        label_index[i] = index

    label_index = np.array(label_index, dtype= int)
    label_index = label_index.astype(np.int32)
    return label_index

train_data, test_data, train_mesh, test_mesh = train_test_split(data, mesh_label, test_size=0.1, random_state = 8)

test_labels = mlb.fit_transform(test_mesh)
test_labelsIndex = getLabelIndex(test_labels)

nb_validation_samples = int(VALIDATION_SPLIT * len(train_data))

x_train = train_data[:-nb_validation_samples]
y_train = train_mesh[:-nb_validation_samples]
x_val = train_data[-nb_validation_samples:]
y_val = train_mesh[-nb_validation_samples:]

def data_generator(input_x, input_y, batch_size = BATCH_SIZE, padding_size = MAX_SEQUENCE_LENGTH):
    
    def padding(input_data,maxlen):
        padded_data = sequence.pad_sequences(input_data, maxlen)
        return padded_data
    input_y_labels = mlb.fit_transform(input_y)
    loopcount = len(input_x) // batch_size
    while True:
        i = random.randint(0, loopcount - 1)
        if len(input_x) == len(input_y_labels):
#            for i in range(0, len(input_x), batch_size):
            x_batch = padding(input_x[i*batch_size:(i+1)*batch_size], MAX_SEQUENCE_LENGTH)
            y_batch = input_y_labels[i*batch_size:(i+1)*batch_size]
            yield x_batch, y_batch
        else:
            print("Input dimension does not match!")

########## use pre-trained word2vec (200d) for embeddings ##########
# read file 
with open ("types.txt", "r") as word:
    word_list = word.readlines()

vectors = open ("vectors.txt", "r")
vector = []
for line in vectors:
    vector.append(line)
vectors.close()

embeddings_index = {}
for i, word in enumerate(word_list):
    token = vector[i].strip()
    token = token.split()
    coefs = np.asarray(token, dtype='float32')
    embeddings_index[word] = coefs
    
EMBEDDING_DIM = 200

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector    
        
########## Training ##########
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable = False)
 
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

convs = []
filter_sizes = [3, 4, 5]

for fsz in filter_sizes:
    l_conv = Conv1D(nb_filter=128,filter_length=fsz,activation='relu')(embedded_sequences)
    l_norm = BatchNormalization()(l_conv)
    pool_size = MAX_SEQUENCE_LENGTH - fsz + 1
    l_pool = MaxPooling1D(pool_size)(l_norm)
    convs.append(l_pool)
    
l_merge = Concatenate(axis = 1)(convs)
l_flat = Flatten()(l_merge)
#l_dense = Dense(128, activation='relu')(l_flat)
#l_norm2 = BatchNormalization()(l_dense)
l_dropout = Dropout(0.5)(l_flat)
preds = Dense(label_dim, activation='sigmoid')(l_dropout)

model = Model(sequence_input, preds)
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['top_k_categorical_accuracy'])

print("model fitting - more complex convolutional neural network")
model.summary()

model.fit_generator(generator = data_generator(x_train, y_train, batch_size = BATCH_SIZE, padding_size = MAX_SEQUENCE_LENGTH),
                    steps_per_epoch = len(x_train) // BATCH_SIZE, epochs = 2, 
                    validation_data = data_generator(x_val, y_val, batch_size = BATCH_SIZE, padding_size = MAX_SEQUENCE_LENGTH),
                    validation_steps = len(x_val) // BATCH_SIZE)

############################### Testing ###################################
test_data = sequence.pad_sequences(test_data, maxlen = MAX_SEQUENCE_LENGTH)
pred = model.predict(test_data)
pred_file = open("CNN_L_Full.pkl", 'wb')
pickle.dump(pred, pred_file)

############################### Evaluations ###################################
# predicted binary labels 
# find the top k labels in the predicted label set
def top_k_predicted(predictions, k):
    predicted_label = np.zeros(predictions.shape)
    for i in range(len(predictions)):
        top_k_index = (predictions[i].argsort()[-k:][::-1]).tolist()
        for j in top_k_index:
            predicted_label[i][j] = 1
    predicted_label = predicted_label.astype(np.int64)
    return predicted_label

top_10_pred = top_k_predicted(pred, 10)
end = time.time()
print("Run Time: ", end - start)
########################### Evaluation Metrics  #############################
# precision @k
precision = precision_at_ks(pred, test_labelsIndex, ks = [1, 3, 5])

for k, p in zip([1, 3, 5], precision):
        print('p@{}: {:.5f}'.format(k, p))

# nDCG @k
nDCG_1 = []
nDCG_3 = []
nDCG_5 = []
Hamming_loss = []
for i in range(pred.shape[0]):
    
    ndcg1 = ndcg_score(test_labels[i], pred[i], k = 1, gains="linear")
    ndcg3 = ndcg_score(test_labels[i], pred[i], k = 3, gains="linear")
    ndcg5 = ndcg_score(test_labels[i], pred[i], k = 5, gains="linear")
    
    hl = hamming_loss(test_labels[0], top_10_pred[0])
    
    nDCG_1.append(ndcg1)
    nDCG_3.append(ndcg3)
    nDCG_5.append(ndcg5)
    
    Hamming_loss.append(hl)

nDCG_1 = np.mean(nDCG_1)
nDCG_3 = np.mean(nDCG_3)
nDCG_5 = np.mean(nDCG_5)
Hamming_loss = np.mean(Hamming_loss)

print("ndcg@1: ", nDCG_1)
print("ndcg@3: ", nDCG_3)
print("ndcg@5: ", nDCG_5)
print("Hamming Loss: ", Hamming_loss)

###### example-based evaluation
# convert binary label back to orginal ones
top_10_labels = mlb.inverse_transform(top_10_pred)
top_20_labels = mlb.inverse_transform(top_20_pred)

# calculate example-based evaluation
example_based_measure_10 = example_based_evaluation(test_mesh, top_10_labels)
print("EMP@10, EMR@10, EMF@10")
for em in example_based_measure_10:
    print(em, ",")

example_based_measure_20 = example_based_evaluation(test_mesh, top_20_labels, 20)
print("EMP@20, EMR@20, EMF@20")
for em in example_based_measure_20:
    print(em, ",")    

# label-based evaluation
label_measure_10 = perf_measure(test_labels, top_10_pred)
print("MaP@10, MiP@10, MaF@10, MiF@10: " )
for measure in label_measure_10:
    print(measure, ",")

label_measure_20 = perf_measure(test_labels, top_20_pred)
print("MaP@20, MiP@20, MaF@20, MiF@20: " )
for measure in label_measure_20:
    print(measure, ",")    
   
print("Finish!")
