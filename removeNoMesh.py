#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 20:23:56 2019

@author: wangxindi
"""
import numpy as np

with open("FullArticle_Large.txt", "r") as data_token:
    datatoken = data_token.readlines()
    
with open("AbstractAndTitle1.txt", "r") as ab_token:
    abtoken = ab_token.readlines()

with open("FullArticle_CaptionAndPara.txt", "r") as cp_token:
    cptoken = cp_token.readlines()
    
# read full mesh list 
with open("MeshIDListFull.txt", "r") as ml:
    meshList = ml.readlines()
print("Lable dimension: ", len(meshList))

mesh_out = []
for mesh in meshList:
    mesh_term = mesh.lstrip('0123456789| .-')
    mesh_term = mesh_term.split("|")
    mesh_term = [ids.strip() for ids in mesh_term]
    mesh_out.append(mesh_term)

tag = "D000000"
NoMesh = []
for i, item in enumerate(mesh_out):
    if tag in item:
        NoMesh.append(i)
        
new_meshList = np.delete(meshList, NoMesh).tolist()
new_mesh = open("MeshIDListLarge.txt", "w", encoding='utf-8')
for line in new_meshList:
    new_mesh.write(line.strip() + "\r")   
new_mesh.close()  

new_abtoken = []
for i, line in enumerate(abtoken):
    if i in NoMesh:
        continue
    else:
        new_abtoken.append(line)
new_ab = open("AbstractAndTitle_Large.txt", "w", encoding='utf-8')
for line in new_abtoken:
    new_ab.write(line.strip() + "\r")   
new_ab.close() 
       
new_cptoken = []
for i, line in enumerate(cptoken):
    if i in NoMesh:
        continue
    else:
        new_cptoken.append(line)        
new_cp = open("CaptionAndPara_Large.txt", "w", encoding='utf-8')
for line in new_cptoken:
    new_cp.write(line.strip() + "\r")   
new_cp.close() 
        
new_datatoken = []
for i, line in enumerate(datatoken):
    if i in NoMesh:
        continue
    else:
        new_datatoken.append(line)  
new_data = open("Full_Large.txt", "w", encoding='utf-8')
for line in new_datatoken:
    new_data.write(line.strip() + "\r")   
new_data.close() 

print("finish!")
