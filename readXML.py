#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 14:37:10 2018

@author: wangxindi
"""

import os
from lxml import html
from lxml.etree import tostring
import re

path1 = "/Users/PMC002XXXXXX.xml/PMC0022XXXXX"
dirs1 = os.listdir(path1)
#dirs5.remove(".DS_Store")

noPMID = []
pmid_out = []
title_out = []
abstract_out = []
caption_out = [] 
paragraph_out = []
fileNumber = 0

for file in dirs1: #traverse 
    if not os.path.isdir(file): #whether file is a directary or not
        xmlfile = open(path1+"/"+file, "r"); #open file
        text = xmlfile.read()
        xmlfile.close()
          
        tree = html.fromstring(text)    
        pmid_xpath = "//article-id[contains(@pub-id-type,'pmid')]/text()" 
        label_caption_xpath = '//fig/caption/p'
        paragraphs_xpath = '//xref[@ref-type="fig"]/parent::p'
    
        pmid_nodes = tree.xpath(pmid_xpath)
        if not pmid_nodes[0]:
           noPMID.append(file)
        else:
            pmid_out.append(str(pmid_nodes[0]))
    
        caption_nodes = tree.xpath(label_caption_xpath)
        captions = []
        for node in caption_nodes:
        #add cleaning text of the node
        #node_text = node.text_content()
            captions.append(tostring(node, encoding='unicode'))

        paragraph_nodes = tree.xpath(paragraphs_xpath)
        paragraphs = [tostring(node, encoding='unicode') for node in paragraph_nodes]
    
        # store captions and paragraphs into lists
        # PMID | Captions    
        for i in range (0, len(captions)):
            figure_caption = str(pmid_nodes[0]).strip() + "| "+ captions[i]
            # delete the superindex
            figure_caption = re.sub("<sup>.*?</sup>","", figure_caption)
            # delete the inner tags in captions
            figure_caption = re.sub("<.*?>", "", figure_caption)
            # remove extra space/ newlines in paragraphs
            figure_caption = re.sub("\n+", " ", figure_caption)
            caption_out.append(figure_caption)
        
        # PMID | paragraph
        for i in range (0, len(paragraphs)):
            figure_paragraph = str(pmid_nodes[0]).strip() + "| "+ paragraphs[i]
            # delete the superindex
            figure_paragraph = re.sub("<sup>.*?</sup>","", figure_paragraph)
            # delete the inner tags in paragraphs
            figure_paragraph = re.sub("<.*?>", "", figure_paragraph)
            # remove extra space/ newlines in paragraphs
            figure_paragraph = re.sub("\n+", " ", figure_paragraph)
            paragraph_out.append(figure_paragraph)
        fileNumber += 1
        print ("File #", fileNumber, "compelted!", "total:", len(dirs1))

# write pmid, title, abstract, caption and paragraphs into different files
pmid_outfile = open("PMID_new1.txt", "w", encoding='utf-8')
for i in range (0, len(pmid_out)):
    pmid_outfile.write(pmid_out[i].strip() + "\r")   
pmid_outfile.close()

caption_outfile = open("Captions1.txt", "w", encoding='utf-8')
for i in range (0, len(caption_out)):
    caption_outfile.write(caption_out[i].strip() + "\r")   
caption_outfile.close()


paragraph_outfile = open("Paragraphs1.txt", "w", encoding='utf-8')
for i in range (0, len(paragraph_out)):
    paragraph_outfile.writelines(paragraph_out[i].strip() + "\r")
paragraph_outfile.close()

noPMID_outfile = open("noPMID.txt", "w", encoding='utf-8')
for i in range (0, len(noPMID)):
    noPMID_outfile.write(noPMID[i].strip() + "\r")   
noPMID_outfile.close()
