# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PIL import ImageDraw
import numpy as np
import cv2
import math
import time
def get_region_pin(labelname,slide,dims):
    '''
    get landmark of image  ,ndpviewstate ,npdi.ndpa
    params:
        labelname : the path of label
        slide : the class OpenSlide that open by the openslide
        dims : which level image we need
    return:
        regions: a list of the regions of landmark, coordinates in the dims-level
        pins: a list of the pins of landmark, coordinates in the dims-level
        colorlist: a list that contains the color of all the region should be
    '''
    tree = ET.parse(labelname)
    labellist=tree.findall('.//annotation')#.//ndpviewstate
    downsamples=slide.level_downsamples[dims]
    mppx=float(slide.properties['openslide.mpp-x'])
    mppy=float(slide.properties['openslide.mpp-y'])
    XOffsetFromSlideCentre=float(slide.properties['hamamatsu.XOffsetFromSlideCentre'])
    YOffsetFromSlideCentre=float(slide.properties['hamamatsu.YOffsetFromSlideCentre'])
#    X=int((XOffsetFromSlideCentre)/(mppx*1000*downsamples))
#    Y=int((YOffsetFromSlideCentre)/(mppy*1000*downsamples))
    h,w=slide.level_dimensions[0]
    colorlist=[]
    regions=[]
    pinlist=[]
    for id in labellist:
        type=dict(id.attrib)['type'] 
        color=dict(id.attrib)['color']
        if type=='freehand':
            pointlist=[]
            points=id.findall('.//pointlist/point')
            for pt in points:
                x=float(pt.findtext('x'))
                y=float(pt.findtext('y'))
                xx=((x-XOffsetFromSlideCentre)/(mppx*1000))+h/2
                yy=((y-YOffsetFromSlideCentre)/(mppy*1000))+w/2
                xx=xx/downsamples
                yy=yy/downsamples
                point=(xx,yy)
                pointlist.append(point)
            colorlist.append(color)
#            k = np.array(pointlist, np.int64).reshape((-1, 1, 2))
#            cv2.polylines(img,k,True,[0,0,255],5)
            regions.append(pointlist)       
        if type=='pin':
            x=float(id.findtext('x'))
            y=float(id.findtext('y'))
            xx=((x-XOffsetFromSlideCentre)/(mppx*1000))+h/2
            yy=((y-YOffsetFromSlideCentre)/(mppy*1000))+w/2
            xx=xx/downsamples
            yy=yy/downsamples
            pin=(xx,yy)
            pinlist.append(pin)
    return regions,pinlist,colorlist
def get_newlabel(regions,pinlist,M):
    '''
    get the new coordinates of landmark after register/transform
    params:
        regions: coordinates, list of regions before register
        pins: coordinates, list of pins before register
        M: the transform martix of register'
    return:
        newregions: coordinates, list of regions after transform
        newpins: coordinates, list of pins, after transform
    '''
    newregions=[]
    newpins=[]
    for region in regions:
        newpoint=[]
        for point in region:
            matrix=np.matrix([point[0],point[1],1.0]).T
            ds=M*matrix
            pt=(float(ds[0]),float(ds[1]))
            newpoint.append(pt)
        newregions.append(newpoint)
    for pin in pinlist:
        m=np.matrix([pin[0],pin[1],1.0]).T
        ds=M*m
        pt=(float(ds[0]),float(ds[1]))
        newpins.append(pt)
    return newregions,newpins
    
def draw(im,regions,pinlist,colorlist):
    '''
    draw the label in the image according to the definition of color in the list
    params:
        im: the Image class image
        regions: coordinate, list of regions 
        pins: coordinate, list of pins 
        colorlist: a list that contains the color of all the regions should be
    return:
        img2: a array ,image that drawn labels
    '''
    draw = ImageDraw.Draw(im) 
    for re in range(len(regions)):
        pointlist=regions[re]
        color=colorlist[re]
        draw.line(pointlist,fill=color,width=5)
    img2=np.asarray(im)
    for pin in pinlist:
        center=(int(pin[0]),int(pin[1]))
        cv2.circle(img2,center,4,(0,0,255),-1)
#    plt.rcParams['figure.figsize'] = 15, 15
#    plt.imshow(img2)
#    plt.show()
    return img2
def get_mask(image,mod):
    '''
    get the mask for the connected domain of the image
    params:
        image: a array,image BGR
        mod: return which kind mask total:totalmask,slice:masklist
    return:
        totalmask: the mask of image
        masklist: the mask list of connected domains
        cnt_list: the coordinates of the contours of connected domains
    '''
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#    thre=np.mean(image)
#    _,I1=cv2.threshold(image, thre, 255,type=cv2.THRESH_BINARY)
    ret,I1=cv2.threshold(image,0,255,cv2.THRESH_OTSU)
    bw=255-I1
    if mod=='nofill':
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        bw = cv2.erode(bw, kernel1)
        return bw
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    bw = cv2.dilate(bw, kernel1)
    h,w=bw.shape[:2]
    mask=np.zeros((h+2,w+2),np.uint8)
    im_floodfill = bw.copy()
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    bw = bw | im_floodfill_inv
    bw, contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contourlist = []
    cnt_list=[]
    area_list=[]
    masklist=[]
    for i in range(len(contours)):
        area=cv2.contourArea(contours[i])
        if area>bw.shape[0]*bw.shape[1]*0.02:
            contourlist.append(contours[i])
            area_list.append(area)
    contourindex=np.arange(len(contourlist))
    totalmask=np.array(np.zeros(bw.shape),dtype=np.uint8)
    cv2.drawContours(totalmask,contourlist,-1,255,-1)
    dic=dict(zip(contourindex,area_list))  
    dic=sorted(dic,key=dic.__getitem__)
    for d in dic:
        img=np.zeros(bw.shape,np.uint8)
        cnt=[]
        cnt.append(contourlist[d])
        cnt_list.append(contourlist[d])
        cv2.drawContours(img,cnt,-1,255,-1)
        masklist.append(img)
    if mod=='slice':
        return masklist,cnt_list
    if mod=='total':
        return totalmask,cnt_list
    
def getmainmask(masklist,idx_region):
    '''
    get the mask that the most landmark is in
    params: 
        masklist: all the mask
        idx_region: which mask the region is in
    return: 
        mask: the mask that the most landmark in 
    '''
    repet_idx=[]
    for i in range(len(masklist)):
        repet_idx.append(idx_region.count(i))
    argmax_idx=np.argmax(repet_idx)
    mask1=masklist[argmax_idx]
    return mask1
def mask_reduce_region(mask1,position,length,downsample):
    '''
    cut the regions provided in the mask
    params:
        mask1: a mask image to operation
        position: a list of the coordinate(x,y) and the size (w,h) of area in the same downsample of mask1
        length: the patch length in the 0-level
        downsample: minification the mask1 is
    return:
        mask: new mask that cut the regions provided and the patches that contains the regions
    '''
    mask2=np.array(np.zeros((mask1.shape)),dtype=np.uint8)
    slength=int(length/downsample)+1
    for pt in position:
        x=int(pt[0])
        y=int(pt[1])
        w=int(pt[2])
        h=int(pt[3])
        if max(x,y)>max(mask1.shape[0],mask1.shape[1]):
            return 'Error: the position is not fit mask,position and mask must in the same downsample'
        cv2.rectangle(mask2,(x,y),(x+w,y+h),255,-1) 
        cv2.rectangle(mask2,(x-slength,y),(x,y+slength),200,-1) 
        cv2.rectangle(mask2,(x,y-slength),(x+slength,y),200,-1) 
        cv2.rectangle(mask2,(x-slength,y-slength),(x,y),200,-1) 
    mask=mask1-mask2
    return mask

def judge_label(regions,pins,contours,returnnew=None):
    '''
    according to the contours to judging which mask/contour that region/pin is in
    params:
        regions:the list of region,coordinates
        pins:the list of pin, coordinates
        contours:the list of contour coordinates
        returnnew: is None return idx,idxpin
                   is True return new_regions_con,new_pins_con,idx,idxpin
   return:
       new_regions_con: shape (-1,len(contour)),the new list of regions that sorted accord contour
       new_pins_con: shape (-1,len(contour)),the new list of pins that sorted accord contour
       idx: index of contours that region is , shape like regions
       idxpin: index of contours that pin is ,shape like pins
    '''
    new_regions_con=[[]]*len(contours)
    new_pins_con=[[]]*len(contours)
    idx=[]
    idxpin=[]
    for region in regions:
        ran=np.random.randint(0,len(region),1)[0]
        point=region[ran]
        for id in range(len(contours)):
            ass=cv2.pointPolygonTest(contours[id],point,True)
            if ass>=0:
                new_regions_con.insert(id,region)
                idx.append(id)
    for pin in pins:
        for id in range(len(contours)):
            ass=cv2.pointPolygonTest(contours[id],point,True)
            if ass >=0:
                new_pins_con.insert(id,pin)
                idxpin.append(id)
    if returnnew==True:
        return new_regions_con,new_pins_con,idx,idxpin
    else:
        return idx,idxpin
   
def get_newlabel_num(regions,pins,idx_region,idx_pin,M_lis):
    '''
    according to which contours the region/pin is in to get the new region/pin
    params: 
        regions: the list of region 
        pins: the list of pin 
        idx_region: which contour region is in 
        idx_pin: which contour pin is in 
        M_lis: a martix list of contours/connected domains
    return:
        newregions: a list that new region after transform
        newpins: a list that new pin after transform
    '''
    newregions=[]
    newpins=[]
#    M2_=np.vstack((M_warp,[0,0,1]))
    for re in range(len(regions)):
        region=regions[re]
        id=idx_region[re]
        M3=M_lis[id]
#        M1=np.vstack((M,[0,0,1]))
#        M3=np.dot(M1,M2_)
        newpoint=[]
        for point in region:
            matrix=np.matrix([point[0],point[1],1.0]).T
            ds=M3*matrix
            pt=(float(ds[0]),float(ds[1]))
            newpoint.append(pt)
        newregions.append(newpoint)
    for p in range(len(pins)):
        pin=pins[p]
        id=idx_pin[p]
        M3=M_lis[id]
#        M1=np.vstack((M,[0,0,1]))
#        M3=np.dot(M1,M2_)
        m=np.matrix([pin[0],pin[1],1.0]).T
        ds=M3*m
        pt=(float(ds[0]),float(ds[1]))
        newpins.append(pt)
    return newregions,newpins
