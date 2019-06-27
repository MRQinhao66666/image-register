import time
import os 
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import xml_draw as xml
import slice_utils
import openslide as opsl
import glob
# =============================================================================
current_path = os.getcwd()
###   get path of image and label
image_path=os.path.join(current_path,'image')
label_path=os.path.join(current_path,'label')
###   define params 
withcontour=None
length=300   
dim=4 
num=50
###   define the path result to save
result_path =os.path.join(current_path,'slice',str(withcontour)+'-square'+str(length))
if not os.path.exists(result_path):
    os.makedirs(result_path)
###   get file list in floder
file =glob.glob(os.path.join(image_path,'*HE.ndpi'))
for idx in file:
    start_time1 = time.time()
###   get the file name
    base=os.path.basename(idx)
###   get the total path of image
    file_source_path=os.path.join(image_path,str(idx))
###   get the total path of label
    label_name=os.path.join(label_path,str(base)[:-8]+'.ndpi.ndpa')
###   use opsl get image in the dim that define
    slide1=opsl.OpenSlide(file_source_path)
    downsample=slide1.level_downsamples[dim]
    print(slide1.level_downsamples)
    slide_thumbnail1 = slide1.get_thumbnail(slide1.level_dimensions[dim])
    image=np.array(slide_thumbnail1)
###   get the masklist and contours of different connected domains
    masklist,cnts=xml.get_mask(image,'slice')
###   get the coordinate of regions and pins in the dim-level
    regions1,pins1,colorlist=xml.get_region_pin(label_name,slide1,dim)
###   get the total position of region that in the dim-level
    position_all=slice_utils.getposition_area(regions1)
###   get the main mask of landmark
    idx_region,idx_pin=xml.judge_label(regions1,pins1,cnts)
    mask1=xml.getmainmask(masklist,idx_region)
###   get the idx of green landmark and yellow landmark
    greenlist=[]
    yellowlist=[]
    for index,nums in enumerate(colorlist):
        if nums=='#00ff00':
            greenlist.append(index)
        if nums=='#ffff00':
            yellowlist.append(index)
###   the coordinate of regions and pins in the 0-level
    regions,pins,colorlist=xml.get_region_pin(label_name,slide1,0)
###   according the coordinate of regions in the 0-level and idx of green landmark to cut slice that size like (length,length)
    for gidx in greenlist:
        print('green')
        save_path=os.path.join(result_path,'green')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        region=regions[gidx]
# =============================================================================
# ###   get rectangle of regions that landmark green 
#         slice_utils.cutrectangle(region,slide1,save_path,base,gidx,withcontour)
# =============================================================================
        position,greenimage_num=slice_utils.cutregion(region,slide1,length,save_path,base,gidx,withcontour)
###   according the coordinate of regions in the 0-level and idx of yellow landmark to cut the total area with rectangle
    for yidx in yellowlist:
        print('yellow')
        save_path1=os.path.join(result_path,'yellow')
        if not os.path.exists(save_path1):
            os.makedirs(save_path1)
        region=regions[yidx]
        slice_utils.cutrectangle(region,slide1,save_path1,base,yidx,withcontour)
###   get the mask that reduce the region area
    mask=xml.mask_reduce_region(mask1,position_all,length,downsample)
###   cut and save randome image
    save_path2=os.path.join(result_path,'random'+str(num))
    if not os.path.exists(save_path2):
        os.makedirs(save_path2)
    mask3=np.zeros(mask.shape,np.uint8)
    slice.utils.cutrandomimage(mask,downsample,length,slide1,base,save_path,num)
    endtime=time.time()
    print(endtime-start_time1)
print("end")