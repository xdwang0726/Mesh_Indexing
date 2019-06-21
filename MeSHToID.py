#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 01:28:09 2018

@author: wangxindi

Mapping Mesh terms to ID
"""
mapping_id = {}
with open('MeSH_name_id_mapping_2018.txt') as f:
    for line in f:
        (key, value) = line.split('=')
        mapping_id[key] = value
        
with open("bioASQ_processedMeSH.txt", "r") as ml:
    meshList = ml.readlines()


    
mesh_id = []
for mesh in meshList:
    new_term = []
    mesh = mesh.split("|")
    for i in range(1, len(mesh)):
        index = mapping_id.get(mesh[i].strip())
        new_term.append(index)
    new_term.insert(0, mesh[0])
    mesh_id.append(new_term)

mesh_id_list = []
for mesh in mesh_id:
    new_mesh = []
    mesh = [x for x in mesh if x != None]
    for ids in mesh:
        ids = ids.strip()
        new_mesh.append(ids)
    mesh_id_list.append(new_mesh)



id_list = []
for i in range(0, len(mesh_id_list)):
    new_line = "|".join(term.strip() for term in mesh_id_list[i])
    id_list.append(new_line)
        
    
id_list_file = open("MeshIDList_bioASQ.txt", "w", encoding='utf-8')
for i in range(0, len(id_list)):
    id_list_file.write(id_list[i] + "\n")
id_list_file.close()
