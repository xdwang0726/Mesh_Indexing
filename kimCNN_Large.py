#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 17:05:09 2018

@author: wangxindi

"""

"""
kimCNN for large dataset
"""

import numpy as np
from data_helper import text_preprocess

from keras.preprocessing import sequence

########## FIXED PARAMETERS ##########
BATCH_SIZE = 10
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2

########## Data preprocess ##########
with open("FullArticle_Large.txt", "r") as data_token:
    datatoken = data_token.readlines()

datatoken = list(filter(None, datatoken))
print('Total number of document: %s' % len(datatoken))
    
load_data = []
data_length = len(datatoken)    
for i in range(0, data_length):
    token = datatoken[i].lstrip('0123456789.- ')
    token = text_preprocess(token)
    load_data.append(token)
    
from keras.preprocessing.text import Tokenizer
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
# padding document to same length

#data = sequence.pad_sequences(x_seq, maxlen = MAX_SEQUENCE_LENGTH)
#print('Shape of data tensor:', data.shape)


from sklearn.preprocessing import MultiLabelBinarizer
import re
# read full mesh list 
with open("MeshList1.txt", "r") as ml:
    meshList = ml.readlines()

def remove_slash_asterisk(mesh):
    new_list = []
    for it in mesh:
        mesh = re.sub("/.*", "", it)
        mesh = mesh.replace("*", "")
        new_list.append(mesh)
    return new_list

mesh_out = []
for mesh in meshList:
    mesh_term = mesh.lstrip('0123456789| .-')
    mesh_term = mesh_term.split("| ")
    mesh_term = remove_slash_asterisk(mesh_term)
    mesh_out.append(mesh_term)

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(mesh_out)
label_dim = labels.shape[1]
print('Shape of label tensor:', labels.shape)

VALIDATION_SPLIT = 0.2
# TRAIN_TEST_SPLIT = 12000
data = []
indices = np.arange(len(x_seq))
np.random.shuffle(indices)
print("Indices length: %s" % len(indices))
for i in indices:
    data.append(x_seq[i])
label = labels[indices]


def getLabelIndex(label):
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

train_data = x_seq[:280000]
test_data = x_seq[280000:]

train_labels = labels[:280000]
test_labels = labels[280000:]
test_labelsIndex = getLabelIndex(test_labels)

nb_validation_samples = int(VALIDATION_SPLIT * len(train_data))

x_train = train_data[:-nb_validation_samples]
y_train = train_labels[:-nb_validation_samples]
x_val = train_data[-nb_validation_samples:]
y_val = train_labels[-nb_validation_samples:]


def data_generator(input_x, input_y, batch_size = BATCH_SIZE, padding_size = MAX_SEQUENCE_LENGTH):
    
    def padding(input_data,maxlen):
        padded_data = sequence.pad_sequences(input_data, maxlen)
        return padded_data
    
    while True:
        if len(input_x) == len(input_y):
            for i in range(0, len(input_x), batch_size):
                x_batch = padding(input_x[i:i + batch_size], MAX_SEQUENCE_LENGTH)
                y_batch = input_y[i:i + batch_size]
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

#word2vec = []
#for i, vec in enumerate(vector):
#    w2v = word_list[i] + " " + vec
#    word2vec.append(w2v)

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
        
from keras.layers import Embedding
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)


########## Training ##########
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
from keras.models import Model


convs = []
filter_sizes = [3,4,5]

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

for fsz in filter_sizes:
    l_conv = Conv1D(nb_filter=128,filter_length=fsz,activation='relu')(embedded_sequences)
    pool_size = MAX_SEQUENCE_LENGTH - fsz + 1
    l_pool = MaxPooling1D(pool_size)(l_conv)
    convs.append(l_pool)
    
l_merge = Merge(mode='concat', concat_axis=1)(convs)
#out = Dropout(0.25)(l_merge)
#l_cov1= Conv1D(128, 5, activation='relu')(l_merge)
#l_pool1 = MaxPooling1D(5)(l_cov1)
#l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
#l_pool2 = MaxPooling1D(30)(l_cov2)
l_flat = Flatten()(l_merge)
l_dense = Dense(128, activation='relu')(l_flat)
#dense_out = Dropout(0.5)(l_dense)
preds = Dense(label_dim, activation='sigmoid')(l_dense)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

print("model fitting - more complex convolutional neural network")
model.summary()

model.fit_generator(generator = data_generator(x_train, y_train, batch_size = BATCH_SIZE, padding_size = MAX_SEQUENCE_LENGTH),
                    steps_per_epoch = int(len(x_train) / BATCH_SIZE), epochs = 20, 
                    validation_data = data_generator(x_val, y_val, batch_size = BATCH_SIZE, padding_size = MAX_SEQUENCE_LENGTH),
                    validation_steps = int(len(x_val) / BATCH_SIZE), workers = 1)
#model.fit(train_data, y_train, validation_data=(val_data, y_val),
#          epochs=20, batch_size=50)

########## Testing & Evaluations ##########
test_data = sequence.pad_sequences(test_data, maxlen = MAX_SEQUENCE_LENGTH)
pred = model.predict(test_data)


from eval_helper import precision_at_ks

prediction = precision_at_ks(pred, test_labelsIndex, ks=[1, 3, 5])

for k, p in zip([1, 3, 5], prediction):
        print('p@{}: {:.5f}'.format(k, p))

print("Finish!")

########## K-fold cross validation ##########
#from sklearn.model_selection import StratifiedKFold

# Instantiate the cross validator
#kfold_splits = 10
#skf = StratifiedKFold(n_splits=kfold_splits, shuffle=True)

#for index, (train_indices, val_indices) in enumerate(skf.split(train_data, train_labels)):
#    print("Training on fold " + str(index+1))
#    # Generate batches from indices
#    xtrain, xval = train_data[train_indices], train_data[val_indices]
#    ytrain, yval = train_labels[train_indices], train_labels[val_indices] 
   
