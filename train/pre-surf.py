#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 01:03:23 2019

@author: root
"""

import numpy as np
import cv2
import register
import feature
import compute
import os
import time
import gc
import matplotlib.pyplot as plt
#import itertools
###    get the time of the begining of all
start_time = time.time()
###   get the path of image/label folder
current_path = os.getcwd()
image_path = os.path.join(current_path, 'all','image')
label_path = os.path.join(current_path, 'all','label')
###   get the result path ,resultfile name ,error path
result_filename = os.path.join(current_path, 'all','newdataset-.csv')
result_filename2 = os.path.join(current_path, 'all','result-.txt')
path=os.path.join(current_path,'all','dataset_medium.csv')
###    define the fixed size of pre-register and the shrink num of feature-register
shrink_num2=5
fixedsize=200
string=[]
###   read the path from the csv file
f_csv=open(path,'r+')
lines=f_csv.readlines()
#def main():
#    for i in range(1,len(lines)):
#error_idx=[12,72,119,256,273,274,319,320,329,330,433,439]
#error_idx=[319,320,329,330]
#lines_error=[]
#for idx in error_idx: 
#    lines_error.append(lines[idx])
#for line in lines_error[:]:
for line in lines[1:]:
#        if "training" in line:
#             result_path =os.path.join(current_path,'all','result-all','training')
#        if "evaluation" in line:
#            result_path =os.path.join(current_path,'all','result-all','evaluation')
    ###   define the result_path to save 
        result_path =os.path.join(current_path,'all','result-')
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        start_time1 = time.time()
        string=str(line).split(',')
###   get the image and label path
        file_source,file_target=string[4],string[6]
        label_source,label_target=string[5],string[7]
        file_source_path=os.path.join(image_path,file_source)
        file_target_path=os.path.join(image_path,file_target)
        label_source_path=os.path.join(label_path,label_source)
        a=str(file_source).split('/')
###   get the name of source image/label,and save path to new label(save_csv2)
        save_name = os.path.join(os.path.basename(file_source)[:-4]+
                                     'to'+os.path.basename(file_target)[:-4])
        sname=os.path.basename(file_source)[:-4]+'.csv'
        saving_path =os.path.join(result_path,'label',string[0])
        save_map= os.path.join(result_path,'image',a[0],a[1])
        if not os.path.exists( saving_path):
                os.makedirs( saving_path)
        if not os.path.exists(save_map):
                os.makedirs(save_map)
        save_csv2=os.path.join(saving_path,sname)
        bstring=str(save_csv2).split('/')
        savestring=bstring[8]+'/'+bstring[9]
#        save_csv1=os.path.join(saving_path1,str(save_name)+'M.txt')
        save_image=os.path.join(save_map,str(save_name))
###   get the image and label
        data_source = (np.loadtxt(label_source_path, dtype=np.str, delimiter=","))[1:, 1:].astype(np.float)
        img_source = cv2.imread(file_source_path,1)
        img_target = cv2.imread(file_target_path,1)
###   pre-register
        pre=register.Pre(img_source,img_target,fixedsize)
        M_warp=pre.pre_register()
        width2=pre.width2
        height2=pre.height2
        img_warp=cv2.warpAffine(img_source, M_warp, (width2, height2))
###   feature register
        feature1=feature.Feature(img_warp,img_target,shrink_num2)
        M2,I,lenmatch,threshold,in_num,Ir=feature1.register()
###   get the shrink image after transform,and the mix of two image
        shrink1=feature1.dimg1
        shrink2=feature1.dimg2
        w2,h2=feature1.width2,feature1.height2
        warp = cv2.warpAffine(shrink1, M2, (w2,h2))
        merge = np.uint8(shrink2* 0.5 + warp * 0.5)
        cv2.imwrite(str(save_image) + '-mix.jpg', merge)
        plt.imshow(merge)
        plt.show()
###   get the transform of the total size        
        M=M2
        M[0,2]=5*M2[0,2]
        M[1,2]=5*M2[1,2]
        M_warp=np.vstack((M_warp,[0,0,1]))
        M2_=np.vstack((M,[0,0,1]))
        M3=np.dot(M2_,M_warp)
###   save new landmark cooridinates and match image
        data_save= compute.get_data(data_source, M3)
        np.savetxt(save_csv2,data_save,fmt=('%s,%s,%s'),delimiter=',')
        del data_save,data_source,warp,merge
        gc.collect()
        cv2.imwrite(str(save_image) + 'register.jpg', I)
        del img_source,img_target,img_warp,I,shrink1,shrink2
        gc.collect()
        end_time1 = time.time()
        print(save_name)
        time_register =(end_time1 - start_time1)/60
###   write string that needed to a new cvs file        
        line=str(line)[:-3]+' '+','+savestring+','+str(time_register)   
        with open(result_filename, 'a', encoding='utf-8') as f1:
                     f1.writelines('{0:<250s}\n'.format(line))
        with open(result_filename2, 'a', encoding='utf-8') as f2:
                     f2.writelines('{0:<65s}{1:<25s}{2:<25s}{3:<15s}{4:<15s}\n'.
                                   format(str(save_name),str(lenmatch),str(threshold),str(in_num),str(time_register)))