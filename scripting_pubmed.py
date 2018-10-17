#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 22:17:56 2018

@author: wangxindi
"""
import urllib.request
from bs4 import BeautifulSoup
import re

# This script is used to automatically download mesh terms from pubmed given 
# associating PMID

proxy_handler = urllib.request.ProxyHandler({'http':'115.32.41.100:80'})
opener = urllib.request.build_opener(proxy_handler)
urllib.request.install_opener(opener)


def scraping_title(output):
    result = []
    TI_indices = [i for i, s in enumerate(output) if 'TI  -' in s]
    PG_indices = [i for i, s in enumerate(output) if 'PG  -' in s]
    LID_indices = [i for i, s in enumerate(output) if 'LID -' in s]
    
    if PG_indices == []:
        end_indices = LID_indices[0] 
    else:
        end_indices = PG_indices[0]
    
    if TI_indices == []:
        result.append(" No title")
        result = "".join(result)
    else:    
        for i in range (TI_indices[0], end_indices):
        #if output[i].startswith("TI  -"):
            result.append(output[i])
        result = "".join(result)
        result = result.replace("TI  -", "")
        # remove the beginning spaces for each line
        result = re.sub("\s+", " ", result)
    return result

def scraping_abstract(output):
    AB_indices = [i for i, s in enumerate(output) if 'AB  -' in s]
    FAU_indices = [i for i, s in enumerate(output) if 'FAU ' in s] 
    CN_indices = [i for i, s in enumerate(output) if 'CN  -' in s]
    LA_indices = [i for i, s in enumerate(output) if 'LA  -' in s]
    if FAU_indices == []:
        if CN_indices == []:
            end_indices = LA_indices[0]
        else:
            end_indices = CN_indices[0] 
    else:
        end_indices = FAU_indices[0]

    result = []
    if AB_indices == []:
        result.append(" NO abstract")
        result = "".join(result)
    else:
         for k in range (AB_indices[0], end_indices):
    #for i in range (0, length):
        #if output[i].startswith("AB  -"):
            result.append(output[k])
         result = "".join(result)
         result = result.replace("AB  -", "")
         # remove the beginning spaces for each line
         result = re.sub("\s+", " ", result)
    return result

def remove_substring_MH(string):
    MH = "MH  -"
    string = string.replace(MH, "")
    return string

def scraping_MH(output):
    result = []
    length = len(output)
    for i in range (0, length):
        if output[i].startswith("MH "):
            #check whether it has a line break
            if output[i+1][0:2] == '  ':
                out = output[i].strip() + " " + output[i+1].strip()
                result.append(out)
            else:
                result.append(output[i])
    new_result = []
    for item in result:
        new_result.append(remove_substring_MH(item))
    mesh = "|".join(new_result)
    # determine whether the document does not has mesh terms
    if not mesh:
        mesh = " NO Mesh"
    return mesh


# read PMID from the file      
with open ("PMID_list.txt","r") as fp: 
    pmid_list = fp.readlines()

# scaping mesh terms using pmid
title = []
abstract = []
mesh_list = []
number = 0

notExistPMID = []
noMesh = []


for pmid in pmid_list:
    url = "https://www.ncbi.nlm.nih.gov/pubmed/?term="+pmid.strip()+"&report=medline&format=text"
    ope  = opener.open(url) # open url
    soup = BeautifulSoup(ope, "html.parser") # pull data out from html
    data = soup.pre.string
    if data is None:
        notExistPMID.append(pmid)
        number += 1
        continue 
    else:
        output = data.splitlines()
        title.append(pmid.strip()+"|".strip()+scraping_title(output))
        abstract.append(pmid.strip()+"|".strip()+scraping_abstract(output))
        mesh_list.append(pmid.strip()+"|".strip()+scraping_MH(output))
    
    number += 1
    print("PMID #", number, "completed! Total:", len(pmid_list))


# write title into file
title_file = open("TitleList9.txt","w") 
num_of_articles = len(title)
for i in range (0, num_of_articles):
    title_file.write(title[i] + "\n")
title_file.close()

# write abstract into file
abstract_file = open("Abstract9.txt","w") 
num_of_articles = len(abstract)
for i in range (0, num_of_articles):
    abstract_file.write(abstract[i] + "\n")
abstract_file.close()

# write mesh into file
mesh_file = open("MeshList9.txt","w") 
num_of_articles = len(mesh_list)
for i in range (0, num_of_articles):
    mesh_file.write(mesh_list[i] + "\n")
mesh_file.close()
