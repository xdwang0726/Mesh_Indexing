#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 12:52:48 2019

@author: wangxindi
"""

# pre-porcessing scraping MeSH
import re

with open ("bioASQ_MeshList.txt", "r") as mesh:
    meshList = mesh.readlines()

mesh_out = []
for mesh in meshList:
    mesh_term = mesh.lstrip()
    mesh_term = mesh_term.split("|")
    mesh_term = [ids.strip() for ids in mesh_term]
    mesh_out.append(mesh_term)


processed_mesh = []
for mesh in mesh_out:
    new_mesh = []
    for string in mesh:
        new_str = re.sub("/.*", "", string)
        new_str = re.sub("(\*+).*", "", new_str)
        new_str.strip()
        new_mesh.append(new_str)
        new_mesh = list(filter(None, new_mesh))
    processed_mesh.append(new_mesh)
    
id_list = []
for i in range(0, len(processed_mesh)):
    new_line = "|".join(term.strip() for term in processed_mesh[i])
    id_list.append(new_line)
        
    
id_list_file = open("bioASQ_processedMeSH.txt", "w", encoding='utf-8')
for i in range(0, len(id_list)):
    id_list_file.write(id_list[i] + "\n")
id_list_file.close()
