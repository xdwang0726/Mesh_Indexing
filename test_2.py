#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 15:16:23 2018

@author: wangxindi
"""

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
#from skmultilearn.problem_transform import BinaryRelevance
#from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from skmultilearn.adapt import MLkNN
import pickle

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
print("tfidf_train:", np.shape(document_term_matrix_ab_title))
tfidf_train = open("tfidf_train.pkl", 'wb')
pickle.dump(document_term_matrix_ab_title, tfidf_train)
tfidf_train.close()
#np.savetxt("train_tfidf.csv", document_term_matrix_ab_title)


### TESTING ###
test_tfidf = tfidf.transform(test_corpus)
document_matrix_test = test_tfidf.toarray()
print("tfidf_test:", np.shape(document_matrix_test))
tfidf_test = open("tfidf_test.pkl", 'wb')
pickle.dump(document_matrix_test, tfidf_test)
tfidf_test.close()
#np.savetxt("test_tfidf.csv", document_matrix_test)


########### doc2vec ##########
vector_size = [1000]

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
    model.save("doc2vec{}d.model".format(ndim))

#### TRAINING ####
# get document vector for different dimensions
#length_train = len(train_corpus)
for ndim in vector_size:
    #model_loaded = gensim.models.Doc2Vec.load("doc2vec{}d.model".format(ndim))
    "train{}d_matrix".format(ndim) = model.docvecs.vectors_docs
    print("doc2vec_train:", np.shape("train{}d_matrix".format(ndim)))
    doc2vec_train = open("doc2vec{}d_train.pkl".format(ndim), 'wb')
    pickle.dump("train{}d_matrix".format(ndim), doc2vec_train)
    doc2vec_train.close()
#    for i, doc in enumerate(train_corpus):
#        doc2vec[i] = model_loaded.infer_vector(doc).reshape(1, ndim)
#        np.savetxt("doc2vec{}d_train.csv".format(ndim), doc2vec)
    

#### TESTING ####
# get document vector for different dimensions
length_test = len(test_corpus) 
for ndim in vector_size:
    model_loaded = gensim.models.Doc2Vec.load("doc2vec{}d_test.model".format(ndim))
    "test{}d_matrix".format(ndim) = np.zeros((length_test,ndim))
    for i, doc in enumerate(test_corpus):
        "test{}d_matrix".format(ndim)[i] = model_loaded.infer_vector(doc).reshape(1, ndim)
    print("doc2vec_test:", np.shape(test_matrix))
    doc2vec_test = open("doc2vec{}d.pkl".format(ndim), 'wb')
    pickle.dump("test{}d_matrix".format(ndim), doc2vec_test)
    doc2vec_test.close()
    
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

print("train_y:", np.shape(train_y))
print("test_y:", np.shape(test_y))


# concatanate training set
for ndim in vector_size:
    "train_{}d".format(ndim) = np.concatenate((document_term_matrix_ab_title, "train{}d_matrix".format(ndim)), axis=1)
    print("train_{}d".format(ndim),np.shape("train_{}d".format(ndim)))    
    # concatanate testing set
    "test_{}d".format(ndim) = np.concatenate((document_matrix_test, "test{}d_matrix".format(ndim)), axis=1)
    print("test_{}d".format(ndim),np.shape("test_{}d".format(ndim)))

classifier2 = MLkNN(k=50)
classifier2.fit(train_10d, train_y)
predictions2 = classifier2.predict(test_10d)
predicted_labels = mlb.inverse_transform(predictions2)
#accuracy2 = accuracy_score(test_y,predictions2)
#print(accuracy2)
# save the predicted labels
pf = open('predictedLabels.pkl', 'wb')
pickle.dump(predicted_labels, pf)
pf.close()

# covert tuples to list
new_predicted = []
for p in predicted_labels:
    new_p = list(p)
    new_predicted.append(new_p)
    
# compare predicted labels and original ones
test_label = mlb.inverse_transform(test_y)
same_labels = []
for i in range (len(test_label)):
    same = list(set(new_predicted[i]).intersection(test_label[i]))
    same_labels.append(same)


# return the probability of correct predicted labels
prob = []
for i in range (len(test_label)):
    p = len(same_labels[i]) / len(test_label[i])
    prob.append(p)
prob_list = open('prob_matrix.pkl', 'wb')
pickle.dump(prob, prob_list)
prob_list.close()

probability = sum(prob)/len(test_label)
print("Probability:", probability)
# variable = pickle.load(open('filename.pkl','rb'))