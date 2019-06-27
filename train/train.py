#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 06:32:20 2019
@author: root
"""
import numpy as np
import cv2
import register
import feature#_no_annotation as feature
import compute as c
import matplotlib.pyplot as plt
#import pycpd
import os
import time
import gc


###    get the time of the begining of all
start_time = time.time()
###   get the path of image/label folder
current_path = os.getcwd()
image_path = os.path.join(current_path, 'training','image')
label_path = os.path.join(current_path, 'training','label')
###   get the result path ,resultfile name ,error path
result_path =os.path.join(current_path,'result','11-ff')
error_path = os.path.join(result_path, 'error')
image_dir1 = sorted(os.listdir(image_path),key=str.lower,reverse=False)[4:]
result_filename = os.path.join(result_path, 'result.txt')
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(error_path):
    os.makedirs(error_path)
###    define the fixed size of pre-register and the shrink num of feature-register
shrink_num=5
fixedsize=200
def main():
    ###    write the name of information
    with open(result_filename, 'a', encoding='utf-8') as f1:
             f1.writelines('{0:<20}{1:<65}{2:<25}{3:<25}{4:<15}{5:<15}{6:<20}\n'.
                           format('dir ','name', 'rTRE(median_rTRE)','match_num','threshold','inliers','times'))
###   get the path of image/label file
    for dir1 in image_dir1:
        print(dir1)     
        image_path1=os.path.join(image_path, dir1)
        image_dir2 = sorted(os.listdir(image_path1))
        readlabel = os.path.join(label_path, dir1)
        m=os.listdir(readlabel)[0]
        for dir2 in image_dir2:
            saving_path = os.path.join(result_path, str(dir1),str(dir2))
            if not os.path.exists(saving_path):
                os.makedirs(saving_path)
            image_path2=os.path.join(image_path1, dir2)
            file_image=os.listdir(image_path2)
            for x in file_image:
                str_img=str(x)[-4:]
            file_target_path = os.path.join(image_path2,str(dir2)+str_img)
            file_target=os.path.basename(file_target_path)
            file = [x for x in file_image if x !=file_target]
            label_target_path=os.path.join(readlabel,m,str(file_target)[:-4]+'.csv')
            for idx in file:
                file_source_path=os.path.join(image_path2,str(idx))
                label_source_path=os.path.join(readlabel,m,str(os.path.basename(idx)[:-4])+'.csv')
###   get the time of the begining of register
                start_time1 = time.time()
###   get the image and the label
                img_source = cv2.imread(file_source_path,1)
                img_target = cv2.imread(file_target_path,1)
                data_source = (np.loadtxt(label_source_path, dtype=np.str, delimiter=","))[1:, 1:].astype(np.float)
                data_target = (np.loadtxt(label_target_path, dtype=np.str, delimiter=","))[1:, 1:].astype(np.float)
###   define path to save
                save_name = os.path.join(os.path.basename(idx)[:-4]+'to'+file_target[:-4])
                save_path = os.path.join(saving_path, save_name)
                error_image=os.path.join(error_path,str(save_name)+str(dir1)+'.jpg')
                save_data_path=os.path.join(saving_path,'datalabel')
                if not os.path.exists(save_data_path):
                    os.makedirs(save_data_path)
                print('load:',time.time()-start_time1)
###   pre-register
                pre=register.Pre(img_source,img_target,fixedsize)
                M_warp=pre.pre_register()
                width2=pre.width2
                height2=pre.height2
                img_warp=cv2.warpAffine(img_source, M_warp, (width2, height2))
                print('pre:',time.time()-start_time1)
###   feature register
                feature1=feature.Feature(img_warp,img_target,shrink_num)
                M,I,lenmatch,threshold,in_num,Ir=feature1.register()
                print('feature:',time.time()-start_time1)
###   get the shrink image after transform,and the mix of two image
                shrink1=feature1.dimg1
                shrink2=feature1.dimg2
                w2,h2=feature1.width2,feature1.height2
                warp = cv2.warpAffine(shrink1, M, (w2, h2))
                merge = np.uint8(shrink2* 0.5 + warp * 0.5)
                plt.imshow(merge)
                plt.show()
###   save images
                cv2.imwrite(str(save_path)+'-merge.jpg',merge)
                cv2.imwrite(str(save_path)+'-warp.jpg', warp)
                cv2.imwrite(str(save_path)+'-match.jpg', Ir)
                cv2.imwrite(str(save_path)+'-register.jpg', I)
### get the transform of the total size
                M1=M
                M1[0,2]=5*M1[0,2]
                M1[1,2]=5*M1[1,2]
                M1_=np.vstack((M1,[0,0,1]))
                M2_=np.vstack((M_warp,[0,0,1]))
                M3=np.dot(M1_,M2_)
#                   np.savetxt(os.path.join(save_M ,str(idx)[:-4]+'.txt'),M1)
#                    data_warp=(f.get_data(data_source,M_warp)[1:,1:]).astype('float')
###   get the rTRE and the new data of landmarks,save data
                rTRE,_,_,data_save= c.get_TRE(data_source, data_target, M3,height2,width2)
                data_save1=c.get_data(data_source,M3)
                np.savetxt(os.path.join(save_data_path,str(idx)[:-4]+'.csv'),data_save,
                           fmt=('%s,%s,%s'),delimiter=',')
                np.savetxt(os.path.join(save_data_path,str(idx)[:-4]+'-1.csv'),data_save1,
                           fmt=('%s,%s,%s'),delimiter=',')
                print('save:',time.time()-start_time1)
                print(rTRE)
###   save image that rTRE is bigger than 0.02
                if rTRE>0.02:
                    cv2.imwrite(str(error_image), merge)
                del data_save,data_source,data_target,#inliers,nkp1_s,nkp1_t
                gc.collect()
                del img_source,img_target,I
                gc.collect()
# =============================================================================
#                     ###   draw the landmarks in the image
#                     for i in range(len(data_source)):
#                         x=data_source[i][0]
#                         y=data_source[i][1]
#                         k=np.array((x,y),np.int32).reshape((-1,1,2))
#                         cv2.polylines(img_target,k,True,(0,255,0),100)
#                     for i in range(len(data_target)):
#                         x=data_target[i][0]
#                         y=data_target[i][1]
#                         k=np.array((x,y),np.int32).reshape((-1,1,2))
#                         cv2.polylines(img_target,k,True,(255,0,0),100)
#                     for i in range(1,len(data_save)):
#                         x3=np.float32(data_save[i][1])
#                         y3=np.float32(data_save[i][2])
#                         k=np.array((x3,y3),np.int32).reshape((-1,1,2))
#                         cv2.polylines(img_target,k,True,(0,0,255),100)
# =============================================================================
                end_time1 = time.time()
                print(idx)
                print(save_name)
                time_register =(end_time1 - start_time1)
###    write some information needed
                with open(result_filename, 'a', encoding='utf-8') as f1:
                    f1.writelines('{0:<20}{1:<65s}{2:<25s}{3:<25s}{4:<15s}{5:<15s}{6:<20s}\n'
                                  .format(str(dir1),str(save_name),str(rTRE),str(lenmatch),
                                          str(threshold),str(in_num),str(time_register)))
#                with open(result_filename, 'a', encoding='utf-8') as f1:
#                    f1.writelines('{0:<20}{1:<65s}{2:<25s}{3:<15s}{4:<15s}{5:<15s}\n'
#                                  .format(str(dir1),str(save_name),str(rTRE),str(err)[:8],str(iteration),str(time_register)))
    end_time = time.time()
    sumtime =(end_time-start_time)
    with open(result_filename, 'a', encoding='utf-8') as f1:
        f1.writelines('{0:<20}{1:<20n}\n'.format('all of time:',sumtime))
    print(result_path)
if __name__ == '__main__':
    main()