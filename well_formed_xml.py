#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 22:13:57 2018

@author: wangxindi
"""
import os
# batch open xml file and delete the last line of each xml file 
with open ("PMC.txt","r") as fp:
    pmcid_list = fp.readlines()
    for pmcid in pmcid_list:
        xmlfile = open(pmcid.strip()+".xml","r")
        xml_lines = xmlfile.readlines()
        new_xml_lines = xml_lines[:-1]
        #xmlfile.flush()
        xmlfile.close()
        os.remove(pmcid.strip()+".xml")
        new_xmlfile = open(pmcid.strip()+".xml","a")
        new_xmlfile.writelines(new_xml_lines)
        #xmlfile.flush()
        new_xmlfile.close()
