#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 23:52:06 2018

@author: wangxindi
"""

with open("Abstract.txt", "r") as abstract_token:
    abstracttoken = abstract_token.readlines()

with open("TitleList.txt", "r") as title_token:
    titletoken = title_token.readlines()
    
with open("CombinedCaptions.txt", "r") as caption_token:
    captionstoken = caption_token.readlines()
    
with open("CombinedParagraphs.txt", "r") as para_token:
    paragraphstoken = para_token.readlines()

    
#IDs = []
minID = 99999999
maxID = 0
for index in range(0,len(abstracttoken)):
    if int(abstracttoken[index][0:8])<minID:
        minID = int(abstracttoken[index][0:8])
    if int(abstracttoken[index][0:8])>maxID:
        maxID = int(abstracttoken[index][0:8])
        
IDArraySize = maxID -  minID +1

result = [""]*IDArraySize
# Abstract 
for index in range(0,len(abstracttoken)):
    newID = int(abstracttoken[index][0:8]) - minID 
    result[newID] = abstracttoken[index]
# title
for index in range(0,len(titletoken)):
    newID = int(titletoken[index][0:8]) - minID 
    result[newID] = result[newID].strip() + " " + titletoken[index][9:].strip()
# captions    
for index in range(0,len(captionstoken)):
    newID = int(captionstoken[index][0:8]) - minID 
    if result[newID]!="":
        result[newID] = result[newID].strip() + " " + captionstoken[index][9:].strip()
 # Paragraph   
for index in range(0,len(paragraphstoken)):
    newID = int(paragraphstoken[index][0:8]) - minID 
    if result[newID]!="":
        result[newID] = result[newID].strip() + " " + paragraphstoken[index][9:].strip()
    

full_article = list(filter(None, result))  

#find the common pmid from full_article and mesh_list
#with open("MeshList.txt", "r") as ml:
#    meshList = ml.readlines()

#list_common = []
#for i, mesh_line in enumerate(meshList):
#    for j, article_line in enumerate(full_article):
#        if mesh_line[0:8] == article_line[0:8]:
#            list_common.append(article_line)

MAX_SEQUENCE_LENGTH = max([len(seq) for seq in full_article])
print("MAX_SEQUENCE_LENGTH:%s" % MAX_SEQUENCE_LENGTH)            

fullArticle = open("FullArticle_Small.txt", "w", encoding='utf-8')
for line in full_article:
    fullArticle.write(line.strip() + "\r")   
fullArticle.close()  

print("Total number of document: %s" % len(full_article))
print("Finish!")
    
    
    
