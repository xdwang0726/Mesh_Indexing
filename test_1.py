#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 17:31:18 2018

@author: wangxindi
"""

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
import numpy as np
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from skmultilearn.adapt import MLkNN


########## TFIDF ##########
# read tokenized abstarct, title, captions, paragraphs
with open("output_abstract.txt", "r") as ab_token:
    abtoken = ab_token.readlines()

with open("output_title.txt", "r") as title_token:
    titletoken = title_token.readlines()
    
with open("PMID.txt", "r") as pmid:
    pmid_list = pmid.readlines()

if len(abtoken) == len(titletoken):
    print(len(abtoken))
else:
    print("ARLERT!!")


ab_title = []
ab_length = len(abtoken)    
for i in range(0, ab_length):
    abtoken[i] = abtoken[i].lstrip('0123456789.- ')
    titletoken[i] = titletoken[i].lstrip('0123456789.- ')
    abANDtitle = abtoken[i] + titletoken[i]
    ab_title.append(abANDtitle)

# get training and testing set 
train_corpus = ab_title[:9000]
test_corpus = ab_title[9000:]

### TRAINING ###
# get occurance of unigram 
freq_unigram = nltk.FreqDist()
for item in train_corpus:
    freq_unigram.update(item.split())

# get occurance of bigram
freq_bigram = nltk.FreqDist()
for item in train_corpus:
    item2 = nltk.bigrams(item.split())
    freq_bigram.update(item2)

# find unigrams/bigrams with at least 6 occurances
unigram_more_than_6 = list(filter(lambda x: x[1] >= 6, freq_unigram.items()))
bigram_more_than_6 = list(filter(lambda x: x[1] >= 6, freq_bigram.items()))


# estimate whether string is a number(includes float, negative numbers)
def is_number(n):
    try:
        float(n)   # Type-casting the string to `float`.
                   # If string is not a valid `float`, 
                   # it'll raise `ValueError` exception
    except ValueError:
        return False
    return True

# remove numbers from unigram
unigram_more_than_6_without_number = []
k = 0
for i in range (len(unigram_more_than_6)):
    if is_number(unigram_more_than_6[i][0]):
        k = k + 1
    else:
        unigram_more_than_6_without_number.append(unigram_more_than_6[i])

# get the vocabulary list
unigram_list = []
bigram_list = []
for unigram in unigram_more_than_6_without_number:
    unigram_key = unigram[0]
    unigram_list.append(unigram_key)
    
for bigram in bigram_more_than_6:
    bigram_key = bigram[0]
    bigram_list.append(bigram_key)

vacabulary_ab_title = unigram_list + bigram_list  
vacabulary_ab_title_dict = {keys: i for i, keys in enumerate(vacabulary_ab_title)}

# tf-idf 
tfidf = TfidfVectorizer(vocabulary = vacabulary_ab_title_dict, ngram_range = (1,2))
tfs = tfidf.fit_transform(train_corpus)
document_term_matrix_ab_title = tfs.toarray()
print(np.shape(document_term_matrix_ab_title))
#np.savetxt("train_tfidf.csv", document_term_matrix_ab_title)


### TESTING ###
test_tfidf = tfidf.transform(test_corpus)
document_matrix_test = test_tfidf.toarray()
print(np.shape(document_matrix_test))
#np.savetxt("test_tfidf.csv", document_matrix_test)


########### doc2vec ##########
vector_size = [10]

# tag list: "doc_pmid"
tag = []
for pmid in pmid_list:
    idNew = "doc_%s" % pmid
    tag.append(idNew)
    
tagged_document = [gensim.models.doc2vec.TaggedDocument(doc, [tag[i]]) for i, doc in enumerate(train_corpus)]
for ndim in vector_size: 
    model = gensim.models.Doc2Vec(dm = 0, min_count = 6, vector_size = ndim, window = 2)   
    model.build_vocab(tagged_document)
    model.train(tagged_document, epochs = 20, total_examples = model.corpus_count)
    #model.save("doc2vec{}d.model".format(ndim))

#### TRAINING ####
# get document vector for different dimensions
#length_train = len(train_corpus)
for ndim in vector_size:
    #model_loaded = gensim.models.Doc2Vec.load("doc2vec{}d.model".format(ndim))
    train_matrix = model.docvecs.vectors_docs
    print(np.shape(train_matrix))
#    for i, doc in enumerate(train_corpus):
#        doc2vec[i] = model_loaded.infer_vector(doc).reshape(1, ndim)
#        np.savetxt("doc2vec{}d_train.csv".format(ndim), doc2vec)
    

#### TESTING ####
# get document vector for different dimensions
length_test = len(test_corpus) 
for ndim in vector_size:
    model_loaded = gensim.models.Doc2Vec.load("doc2vec{}d.model".format(ndim))
    test_matrix = np.zeros((length_test,ndim))
    for i, doc in enumerate(test_corpus):
        test_matrix[i] = model_loaded.infer_vector(doc).reshape(1, ndim)
#        np.savetxt("doc2vec{}d_test.csv".format(ndim), doc2vec)
    print(np.shape(test_matrix))
    
########## Y-LABEL ##########
from sklearn.preprocessing import MultiLabelBinarizer
#import numpy as np
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
y = mlb.fit_transform(mesh_out)

train_y = y[:9000]
test_y = y[9000:]


############# CLASSIFIER1: Binary Relevance ############

classifier1 = BinaryRelevance(GaussianNB())
classifier1.fit(train_10d, train_y)
predictions1 = classifier1.predict(test_10d)
accuracy1 = accuracy_score(test_y,predictions1)
print("binary:" + accuracy1)


############# CLASSIFIER2: ML-KNN ##############
classifier2 = MLkNN(k = 50)
classifier2.fit(train_10d, train_y)
predictions2 = classifier2.predict(test_10d)
accuracy2 = accuracy_score(test_y,predictions2)
print("MLKNN:" + accuracy2)

    
