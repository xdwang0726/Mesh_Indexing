#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 22:37:40 2018

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

from keras.layers import Dense, Input
from keras.layers import Bidirectional, LSTM, Embedding, Concatenate
from keras.optimizers import Adam
from keras.models import Model
from sklearn.metrics import hamming_loss
from eval_helper import precision_at_ks, ndcg_score, perf_measure, example_based_evaluation, hierachy_eval

start = time. time()
#### GPU specified ####
os.environ["CUDA_DEVUCE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

########## FIXED PARAMETERS ##########
BATCH_SIZE = 10
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2

########## Data preprocess ##########
with open("FullArticle_Small.txt", "r") as data_token:
    datatoken = data_token.readlines()
    
with open("AbstractAndTitle.txt", "r") as ab_token:
    abtoken = ab_token.readlines()

with open("Small_CaptionAndPara.txt", "r") as cp_token:
    cptoken = cp_token.readlines()
    
datatoken = list(filter(None, datatoken))
abtoken = list(filter(None, abtoken))
cptoken = list(filter(None, cptoken))
print('Total number of document: %s' % len(datatoken))
    
def preprocess(data):
    load_data = []
    data_length = len(data)    
    for i in range(0, data_length):
        token = data[i].lstrip('0123456789.- ')
        token = text_preprocess(token)
        load_data.append(token)
    return load_data

processed_data = preprocess(datatoken)
processed_ab = preprocess(abtoken)
processed_cp = preprocess(cptoken)
    
tokenizer = Tokenizer()
tokenizer.fit_on_texts(processed_data)

x_seq = tokenizer.texts_to_sequences(processed_data)
ab_seq = tokenizer.texts_to_sequences(processed_ab)
cp_seq = tokenizer.texts_to_sequences(processed_cp)
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
MAX_SEQUENCE_LENGTH_ab = max([len(seq) for seq in ab_seq])
MAX_SEQUENCE_LENGTH_cp = max([len(seq) for seq in cp_seq])
seconde_largest_length_cp= second_largest([len(seq) for seq in cp_seq])
#print("Max sequence length in abstract and title: %s " % MAX_SEQUENCE_LENGTH_ab)
#print("Max sequence length in captions and paragraphs: %s " % MAX_SEQUENCE_LENGTH_cp)
#print("Second largest sequence length: %s" % seconde_largest_length)

if MAX_SEQUENCE_LENGTH_cp >= seconde_largest_length_cp:
    MAX_SEQUENCE_LENGTH_cp = seconde_largest_length_cp
print("abstract and title padding size: %s" % MAX_SEQUENCE_LENGTH_ab)
print("caption and paragraph padding size: %s" % MAX_SEQUENCE_LENGTH_cp)
# read full meshIDs
with open("MeshIDList.txt", "r") as ml:
    meshIDs = ml.readlines()
    
meshIDs = [ids.strip() for ids in meshIDs]
label_dim = len(meshIDs)
mlb = MultiLabelBinarizer(classes = meshIDs)

# read full mesh list 
with open("MeshIDListSmall.txt", "r") as ml:
    meshList = ml.readlines()
print("Lable dimension: ", len(meshList))

mesh_out = []
for mesh in meshList:
    mesh_term = mesh.lstrip('0123456789| .-')
    mesh_term = mesh_term.split("|")
    mesh_term = [ids.strip() for ids in mesh_term]
    mesh_out.append(mesh_term)

# shuffle index
indices = np.arange(len(x_seq))
np.random.shuffle(indices)
print("Indices length: %s" % len(indices))


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
    
train_data_index, test_data_index = train_test_split(indices, test_size=0.1, random_state = 8)
train_data_index = train_data_index.tolist()
test_data_index = test_data_index.tolist()

# get train data
train_data_ab = []
train_data_cp = []
train_mesh = []
for i in train_data_index:
    train_data_ab.append(ab_seq[i])
    train_data_cp.append(cp_seq[i])
    train_mesh.append(mesh_out[i])

test_data_ab = []
test_data_cp = []
test_mesh = []
for i in test_data_index:
    test_data_ab.append(ab_seq[i])
    test_data_cp.append(cp_seq[i])
    test_mesh.append(mesh_out[i])

test_labels = mlb.fit_transform(test_mesh)
test_labelsIndex = getLabelIndex(test_labels)

# save true label into file
#true_label = open('MultichannelTextCNN_true_label.txt', 'w')
#for meshs in test_mesh:
#    mesh = ' '.join(meshs)
#    true_label.writelines(mesh.strip()+ "\r")
#true_label.close()

nb_validation_samples = int(VALIDATION_SPLIT * len(train_data_ab))

x_train_ab = train_data_ab[:-nb_validation_samples]
x_train_cp = train_data_cp[:-nb_validation_samples]
y_train = train_mesh[:-nb_validation_samples]
x_val_ab = train_data_ab[-nb_validation_samples:]
x_val_cp = train_data_cp[-nb_validation_samples:]
y_val = train_mesh[-nb_validation_samples:]

def data_generator_two(input_x1, input_x2, input_y, batch_size, padding_size1, padding_size2):
    def padding(input_data,maxlen):
        padded_data = sequence.pad_sequences(input_data, maxlen)
        return padded_data
    input_y_labels = mlb.fit_transform(input_y)
    loopcount = len(input_x1) // batch_size
    while True:
        i = random.randint(0, loopcount - 1)
        if len(input_x1) == len(input_y_labels) and len(input_x2) == len(input_y_labels):
            x1_batch = padding(input_x1[i*batch_size:(i+1)*batch_size], padding_size1)
            x2_batch = padding(input_x2[i*batch_size:(i+1)*batch_size], padding_size2)
            y_batch = input_y_labels[i*batch_size:(i+1)*batch_size]
            yield [x1_batch, x2_batch], y_batch
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

# channel 1: abstract and title    
embedding_layer_ab = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH_ab,
                            trainable = False)
 
sequence_input_ab = Input(shape=(MAX_SEQUENCE_LENGTH_ab,), dtype='int32')
embedded_sequences_ab = embedding_layer_ab(sequence_input_ab)
l_lstm_ab = Bidirectional(LSTM(200))(embedded_sequences_ab)


# channel 2: captions and paragraphs
embedding_layer_cp = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH_cp,
                            trainable = False)
 
sequence_input_cp = Input(shape=(MAX_SEQUENCE_LENGTH_cp,), dtype='int32')
embedded_sequences_cp = embedding_layer_cp(sequence_input_cp)
l_lstm_cp = Bidirectional(LSTM(200))(embedded_sequences_cp)

l_merge = Concatenate(axis=2)([l_lstm_ab, l_lstm_cp])
preds = Dense(label_dim, activation='sigmoid')(l_merge)

model = Model([sequence_input_ab, sequence_input_cp], preds)
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['top_k_categorical_accuracy'])

print("model fitting - more complex convolutional neural network")
model.summary()

model.fit_generator(generator = data_generator_two(x_train_ab, x_train_cp, y_train, batch_size = BATCH_SIZE, 
                                                   padding_size1 = MAX_SEQUENCE_LENGTH_ab, padding_size2 = MAX_SEQUENCE_LENGTH_cp),
                    steps_per_epoch = len(x_train_ab) // BATCH_SIZE, epochs = 20, 
                    validation_data = data_generator_two(x_val_ab, x_val_cp, y_val, batch_size = BATCH_SIZE, 
                                                   padding_size1 = MAX_SEQUENCE_LENGTH_ab, padding_size2 = MAX_SEQUENCE_LENGTH_cp),
                    validation_steps = len(x_val_ab) // BATCH_SIZE)
############################### Evaluations ###################################
test_data_ab = sequence.pad_sequences(test_data_ab, maxlen = MAX_SEQUENCE_LENGTH_ab)
test_data_cp = sequence.pad_sequences(test_data_cp, maxlen = MAX_SEQUENCE_LENGTH_cp)
pred = model.predict([test_data_ab,test_data_cp])

# predicted binary labels 
# find the top k labels in the predicted label set
def top_k_predicted(goldenTruth, predictions, k):
    predicted_label = np.zeros(predictions.shape)
    for i in range(len(predictions)):
        goldenK = len(goldenTruth[i])
        if goldenK <= k:
            top_k_index = (predictions[i].argsort()[-goldenK:][::-1]).tolist()
        else:
            top_k_index = (predictions[i].argsort()[-k:][::-1]).tolist()
        for j in top_k_index:
            predicted_label[i][j] = 1
    predicted_label = predicted_label.astype(np.int64)
    return predicted_label


top_5_pred = top_k_predicted(test_mesh, pred, 5)
# convert binary label back to orginal ones
top_5_mesh = mlb.inverse_transform(top_5_pred)
top_5_mesh = [list(item) for item in top_5_mesh]
 
top_10_pred = top_k_predicted(test_mesh, pred, 10)
top_10_mesh = mlb.inverse_transform(top_10_pred)
top_10_mesh = [list(item) for item in top_10_mesh]
       
top_15_pred = top_k_predicted(test_mesh, pred, 15)
top_15_mesh = mlb.inverse_transform(top_15_pred)
top_15_mesh = [list(item) for item in top_15_mesh]

end = time.time()
print("Run Time: ", end - start)

# save predicted label into file 
#pred_label_5 = open('TextCNN_pred_label_5.txt', 'w')
#for meshs in top_5_mesh:
#    mesh = ' '.join(meshs)
#    pred_label_5.writelines(mesh.strip()+ "\r")
#pred_label_5.close()

#pred_label_10 = open('TextCNN_pred_label_10.txt', 'w')
#for meshs in top_10_mesh:
#    mesh = ' '.join(meshs)
#    pred_label_10.writelines(mesh.strip()+ "\r")
#pred_label_10.close()

#pred_label_15 = open('TextCNN_pred_label_15.txt', 'w')
#for meshs in top_15_mesh:
#    mesh = ' '.join(meshs)
#    pred_label_15.writelines(mesh.strip()+ "\r")
#pred_label_15.close()
########################### Evaluation Metrics  #############################
# precision @k
precision = precision_at_ks(pred, test_labelsIndex, ks = [1, 3, 5])

for k, p in zip([1, 3, 5], precision):
        print('p@{}: {:.5f}'.format(k, p))

# check how many documents that have mesh terms greater and equal to 10/15
label_row_total = np.sum(test_labels, axis = 1)
index_greater_10 = [index for index, value in enumerate(label_row_total) if value >= 10]
index_greater_15 = [index for index, value in enumerate(label_row_total) if value >= 15]

def get_label_using_index(org_label, index):
    new_label = []
    for i in index:
        new_label.append(org_label[i])
    return new_label

labelIndex_greater_10 = get_label_using_index(test_labelsIndex, index_greater_10)
labelIndex_greater_15 = get_label_using_index(test_labelsIndex, index_greater_15)
pred_10 = np.asarray(get_label_using_index(pred, index_greater_10))
pred_15 = np.asarray(get_label_using_index(pred, index_greater_15))

# precision at 10 and precision at 15
precision_10 = precision_at_ks(pred_10, labelIndex_greater_10, ks = [10])
print("p@10:", precision_10)
precision_15 = precision_at_ks(pred_15, labelIndex_greater_15, ks = [15])
print("p@15:", precision_15)

# nDCG @k
nDCG_1 = []
nDCG_3 = []
nDCG_5 = []
Hamming_loss_5 = []
Hamming_loss_10 = []
Hamming_loss_15 = []
for i in range(pred.shape[0]):
    
    ndcg1 = ndcg_score(test_labels[i], pred[i], k = 1, gains="linear")
    ndcg3 = ndcg_score(test_labels[i], pred[i], k = 3, gains="linear")
    ndcg5 = ndcg_score(test_labels[i], pred[i], k = 5, gains="linear")
    
    hl_5 = hamming_loss(test_labels[0], top_5_pred[0])
    hl_10 = hamming_loss(test_labels[0], top_10_pred[0])
    hl_15 = hamming_loss(test_labels[0], top_15_pred[0])
    
    nDCG_1.append(ndcg1)
    nDCG_3.append(ndcg3)
    nDCG_5.append(ndcg5)
    
    Hamming_loss_5.append(hl_5)
    Hamming_loss_10.append(hl_10)
    Hamming_loss_15.append(hl_15)

nDCG_1 = np.mean(nDCG_1)
nDCG_3 = np.mean(nDCG_3)
nDCG_5 = np.mean(nDCG_5)

Hamming_loss_5 = np.mean(Hamming_loss_5)
Hamming_loss_5 = round(Hamming_loss_5,5)
Hamming_loss_10 = np.mean(Hamming_loss_10)
Hamming_loss_10 = round(Hamming_loss_10,5)
Hamming_loss_15 = np.mean(Hamming_loss_15)
Hamming_loss_15 = round(Hamming_loss_15,5)
      
print("ndcg@1: ", nDCG_1)
print("ndcg@3: ", nDCG_3)
print("ndcg@5: ", nDCG_5)
print("Hamming Loss@5: ", Hamming_loss_5)
print("Hamming Loss@10: ", Hamming_loss_10)
print("Hamming Loss@15: ", Hamming_loss_15)

###### example-based evaluation

# calculate example-based evaluation
example_based_measure_5 = example_based_evaluation(test_mesh, top_5_mesh)
print("EMP@5, EMR@5, EMF@5")
for em in example_based_measure_5:
    print(em, ",")

example_based_measure_10 = example_based_evaluation(test_mesh, top_10_mesh)
print("EMP@10, EMR@10, EMF@10")
for em in example_based_measure_10:
    print(em, ",")

example_based_measure_15 = example_based_evaluation(test_mesh, top_15_mesh)
print("EMP@15, EMR@15, EMF@15")
for em in example_based_measure_15:
    print(em, ",")    

# label-based evaluation
label_measure_5 = perf_measure(test_labels, top_5_pred)
print("MaP@5, MiP@5, MaF@5, MiF@5: " )
for measure in label_measure_5:
    print(measure, ",")    
    
label_measure_10 = perf_measure(test_labels, top_10_pred)
print("MaP@10, MiP@10, MaF@10, MiF@10: " )
for measure in label_measure_10:
    print(measure, ",")

label_measure_15 = perf_measure(test_labels, top_15_pred)
print("MaP@15, MiP@15, MaF@15, MiF@15: " )
for measure in label_measure_15:
    print(measure, ",")    

############ hierachy evaluation ################
hierachy_eval_5_dis1 = hierachy_eval(test_mesh, top_5_mesh, 1)
print("HP_1@5, HR_1@5: " )
for measure in hierachy_eval_5_dis1:
    print(measure, ",")     
hierachy_eval_5_dis2 = hierachy_eval(test_mesh, top_5_mesh, 2)
print("HP_2@5, HR_2@5: " )
for measure in hierachy_eval_5_dis2:
    print(measure, ",") 

    
hierachy_eval_10_dis1 = hierachy_eval(test_mesh, top_10_mesh, 1)
print("HP_1@10, HR_1@10: " )
for measure in hierachy_eval_10_dis1:
    print(measure, ",")     
hierachy_eval_10_dis2 = hierachy_eval(test_mesh, top_10_mesh, 2)
print("HP_2@10, HR_2@10: " )
for measure in hierachy_eval_10_dis2:
    print(measure, ",") 

    
hierachy_eval_15_dis1 = hierachy_eval(test_mesh, top_15_mesh, 1)
print("HP_1@15, HR_1@15: " )
for measure in hierachy_eval_15_dis1:
    print(measure, ",")     
hierachy_eval_15_dis2 = hierachy_eval(test_mesh, top_15_mesh, 2)
print("HP_2@15, HR_2@15: " )
for measure in hierachy_eval_15_dis2:
    print(measure, ",") 


print("Finish!")
