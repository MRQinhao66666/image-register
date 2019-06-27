#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 06:05:21 2019
@author: root
"""
#import sys
import time
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml_draw as xml
import openslide as opsl
from PIL import Image
import glob
import gc
import register 
import feature
###   get the path of image ,label
current_path = os.getcwd()
image_path=os.path.join(current_path,'image')
label_path=os.path.join(current_path,'label')
###   define the dim,the result path,result_file
dims=4
result_path =os.path.join(current_path,'result',str(dims)+'-slice 1-')
if not os.path.exists(result_path):
    os.makedirs(result_path)
result_filename=os.path.join(result_path,'result.txt')
###   get the file list of folder
file =glob.glob(os.path.join(image_path,'*HE.ndpi'))
for idx in file:
    start_time1 = time.time()
### get the name of flie
    base=os.path.basename(idx)
### get the total path of source target label
    file_source_path=os.path.join(image_path,str(idx))
    file_target_path=os.path.join(str(file_source_path)[:-8]+' SR.ndpi')
    label_name=os.path.join(label_path,str(base)[:-8]+'.ndpi.ndpa')
###   read image by opsl,get coordinate of regions,pins
    slide1=opsl.OpenSlide(file_source_path)
    slide2=opsl.OpenSlide(file_target_path)
    slide_thumbnail1 = slide1.get_thumbnail(slide1.level_dimensions[dims])
    slide_thumbnail2 = slide2.get_thumbnail(slide2.level_dimensions[dims])
    img1=np.array(slide_thumbnail1)
    img2=np.array(slide_thumbnail2)
    regions1,pins1,colorlist=xml.get_region_pin(label_name,slide1,dims)
###   draw regions and pins in the image
    imglabel=xml.draw(slide_thumbnail1,regions1,pins1,colorlist)
# =============================================================================
####   pre_register ,total image   
#     pre=register.Pre(img1,img2,200)
#     M_warp=pre.pre_register()
#     width2=pre.width2
#     height2=pre.height2
#     img_warp=cv2.warpAffine(img1, M_warp, (width2, height2))
#     plt.subplot(121),plt.imshow(img_warp)
#     plt.subplot(122),plt.imshow(img2)
#     plt.show()
# =============================================================================
###   get the masklist and contours of connected domains
    mask1s,contours=xml.get_mask(img1,'slice')
    mask2s,contour2s=xml.get_mask(img2,'slice')
###  according contours get the mask idx of regions,pins
    idx_region,idx_pin=xml.judge_label(regions1,pins1,contours)
###  define
    img3=np.zeros(img2.shape)
    M_lis=[] 
    I=np.zeros((img2.shape[0],img2.shape[1]+img2.shape[1],3))
###   every mask in masklist should be register (pre_register and feature register)
    for m  in range(len(mask1s)):
###   get the connected domains from img(RGB) by the mask
        mask1=mask1s[m][:]>0
        img1_i=np.zeros(img1.shape,dtype=np.uint8)
        img2_i=np.zeros(img2.shape,dtype=np.uint8)
        mask2=mask2s[m][:]>0
        for i in range(3):
            img1_i[:,:,i]=img1[:,:,i]*mask1#+(1-mask1)*255
            img2_i[:,:,i]=img2[:,:,i]*mask2#+(1-mask2)*255
###   pre register the connected domains
        pre=register.Pre(img1_i,img2_i,200)
        M_warp=pre.pre_register()
        width2=pre.width2
        height2=pre.height2
        img_warp_i=cv2.warpAffine(img1_i, M_warp, (width2, height2))
        plt.subplot(121),plt.imshow(img_warp_i)
        plt.subplot(122),plt.imshow(img2_i)
        plt.show()
###   get feature and register
        feature1=feature.Feature(img_warp_i,img2_i,1)
###   get the M ,register match image ,string lenmatch,number of interpoints,match image
        M,I_,lenmatch,threshold,interiornum,Ir=feature1.register()
        match_path=os.path.join(result_path,'match')
        match_filename=os.path.join(match_path,str(base)+str(m)+'-r.jpg')
        cv2.imwrite(match_filename, Ir)
#                M=cv2.getRotationMatrix2D((0,0),0,1)
        img3_i=cv2.warpAffine(img_warp_i,M,(width2,height2))
###   get the NGF of image and image after register
        pre2=register.Pre(img3_i,img2_i,200)
        dst1,dst2=pre2.crop_cross()
        distance,ngf=pre2.NGF(dst1,dst2)
###   get the total martix of pre-register and feature register
        M2_=np.vstack((M_warp,[0,0,1]))
        M1_=np.vstack((M,[0,0,1]))
        MI=np.dot(M1_,M2_)
        M3=MI[0:2]
        M_lis.append(MI)
###   get the all the connected domains after register in the  image img3,and match of all the domains in the I
        img3=img3+img3_i*(img3_i[:]<255)
        I=I+I_
###   write some information 
        with open(result_filename, 'a', encoding='utf-8') as f1:
            f1.writelines('{0:<40s}{1:<25s}{2:<10s}{3:<5s}{4:<20s}\n'
                          .format(str(base),str(lenmatch),str(threshold),str(interiornum),str(distance)))
        del img1_i,img2_i,mask1s,mask2,mask1,mask2s,img_warp_i,I_
        gc.collect()
#            plt.imshow(img3)
#            plt.show()
#            M1_=np.vstack((M1,[0,0,1]))
#            M2_=np.vstack((M_warp,[0,0,1]))
#            M3=np.dot(M1_,M2_)
#            np.savetxt(os.path.join(result_path,str(base)[:-5]+'.txt'),M3)
#            regions2,pins2=get_newlabel(regions1,pins1,M3)
###   get the new landmarks after register
    regions2,pins2=xml.get_newlabel_num(regions1,pins1,idx_region,idx_pin,M_lis)
###   draw the new landmark in the warp image and the target image
    img3_im=Image.fromarray(np.uint8(img3))
    img2_im=Image.fromarray(np.uint8(img2))
    img3_label=xml.draw(img3_im,regions2,pins2,colorlist)
    img2_label=xml.draw(img2_im,regions2,pins2,colorlist)
#    img2_label=cv2.cvtColor(img2_label,cv2.COLOR_BGR2RGB)
###   get the mix image
    merge=img3*0.2+img2_label*0.8
###   save image
    save_path=os.path.join(result_path,str(base)[:-5])
    cv2.imwrite(save_path+'warp.jpg',img3_label)
    cv2.imwrite(save_path+'-label.jpg',img2_label)
    cv2.imwrite(str(save_path)+'.jpg',imglabel)
    cv2.imwrite(str(save_path)+'-mix.jpg',merge)
    cv2.imwrite(str(save_path) + '-register.jpg', I)
    del img1,I,img2
    gc.collect()
    end_time1 = time.time()
    print(idx)
    print(base)
    time_register =(end_time1 - start_time1)
    ###   write the angle,rTRE,and center in the picture,and save the picture
    with open(result_filename, 'a', encoding='utf-8') as f1:
        f1.writelines('{0:<40s}{1:<15s}\n'.format(str(base),str(time_register)))
#            with open(result_filename, 'a', encoding='utf-8') as f1:
#                    f1.writelines('{0:<40s}{1:<15s}{2:<10s}{3:<5s}{4:<15s}\n'
#                                  .format(str(base),str(lenmatch),str(threshold),str(len(in_s[0])),str(time_register)))
print("end")
