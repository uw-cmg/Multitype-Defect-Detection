#!/usr/bin/env python
"""
Provides Helps add the class column into the human labelling results

"""
import argparse
import datetime
import glob
import json
import os
import os.path as osp
import sys
import pandas as pd
import numpy as np
#import PIL.Image

parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
parser.add_argument('csv_dir', help='input annotated csv')
parser.add_argument('logs_dir', help='input annotated logs')
parser.add_argument('new', help='input new dir path')
args = parser.parse_args()
csv = args.csv_dir
logs = args.logs_dir

for csv_file in os.listdir(csv):
    fileName = csv_file.split('results')
    log_name = fileName[0] + 'log.txt'
    with open(logs+"/" + log_name) as f:
        lines = f.readlines()
        
    lines = [x.strip() for x in lines]
    defectR = [] # all the defects index
    
    opened_csv = pd.read_csv(args.csv_dir + "/" + csv_file) 
    opened_csv['class'] = ""
    for j in range(len(lines)):
        if lines[j] != '':
            defectR.append(lines[j])
    defect1 = [] # record the 111 loop
    bd = [] #
    defect3 = []
    for j in range(len(defectR)):
        interval = []
        if defectR[j] == '0':
            interval = defectR[j+1].split(" ")
            for k in range(int(interval[0]),int(interval[1])+1):
                defect1.append(k)
        if defectR[j] == '1':
            interval = defectR[j+1].split(" ")
            for k in range(int(interval[0]),int(interval[1])+1):
                bd.append(k)
        if defectR[j] == '2':
            interval = defectR[j+1].split(" ")
            for k in range(int(interval[0]),int(interval[1])+1):
                defect3.append(k)
    for idx, row in opened_csv.iterrows():
        #print(idx)
        if (idx+1) in defect1:
           # print("111")
            opened_csv.at[idx, 'class'] = 0#1
        if (idx+1) in bd:
            opened_csv.at[idx, 'class'] = 1#2
        if (idx+1) in defect3:
            opened_csv.at[idx, 'class'] = 2#3
    
    opened_csv.to_csv(args.new+"/" + csv_file, sep=',')
