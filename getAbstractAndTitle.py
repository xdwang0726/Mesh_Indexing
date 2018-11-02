#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 17:39:07 2018

@author: wangxindi


train_x with abstract and title 

"""
from data_helper import text_preprocess

with open("Abstract.txt", "r") as ab_token:
    abtoken = ab_token.readlines()

with open("TitleList.txt", "r") as title_token:
    titletoken = title_token.readlines()

if len(abtoken) == len(titletoken):    
    ab_title = []
    error_ab_title = []
    ab_length = len(abtoken)    
    for i in range(0, ab_length):
        ab_list = abtoken[i].split()
        title_list = titletoken[i].split()
        if ab_list[0].strip() == title_list[0].strip(): 
            abtoken[i] = abtoken[i].lstrip('0123456789.- ')
            abtoken[i] = abtoken[i].strip("| ")
            abtoken[i] = text_preprocess(abtoken[i])
            titletoken[i] = titletoken[i].lstrip('0123456789.- ')
            titletoken[i] = titletoken[i].strip("| ")
            titletoken[i] = text_preprocess(titletoken[i])
            abANDtitle = titletoken[i] + " " + abtoken[i]
            ab_title.append(abANDtitle)   
        else:
            error = "Abstract: " + ab_list[0] + "Title: " + title_list[0]
            error_ab_title.append(error)  
else:
    print("Abstract:", len(abtoken))
    print("Title:", len(titletoken))
    
abstractANDtitle = open("AbstractAndTitle.txt", "w", encoding='utf-8')
for line in ab_title:
    abstractANDtitle.write(line.strip() + "\r")   
abstractANDtitle.close()