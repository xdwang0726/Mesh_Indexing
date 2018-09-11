#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 17:05:09 2018

@author: wangxindi

"""
import numpy as np
from data_helper import text_preprocess

########## Data preprocess ##########
with open("FullArticle_Small.txt", "r") as data_token:
    datatoken = data_token.readlines()
    
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
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# find the maximum length of document
MAX_SEQUENCE_LENGTH = max([len(seq) for seq in x_seq])
print(MAX_SEQUENCE_LENGTH)
# padding document to same length
from keras.preprocessing import sequence
data = sequence.pad_sequences(x_seq, maxlen = MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)


from sklearn.preprocessing import MultiLabelBinarizer
import re
# read full mesh list 
with open("MeshList.txt", "r") as ml:
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

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

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

train_data = data[:12000]
test_data = data[12000:]

train_labels = labels[:12000]
test_labels = labels[12000:]
test_labelsIndex = getLabelIndex(test_labels)

nb_validation_samples = int(VALIDATION_SPLIT * train_data.shape[0])

x_train = train_data[:-nb_validation_samples]
y_train = train_labels[:-nb_validation_samples]
x_val = train_data[-nb_validation_samples:]
y_val = train_labels[-nb_validation_samples:]


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
                            trainable = False)


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
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=20, batch_size=50)

########## Testing & Evaluations ##########
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
   
