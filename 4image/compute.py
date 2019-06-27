# -*- coding: utf-8 -*-
import numpy as np
import cv2

def get_data(data,M):
    '''
    get the new landmark coordinates after transform
    params:
        data: the coordinates 
        M: transform martix
    return:
        data_warp:the new landmark coordinates after transform
    '''
    data_warp = [[0,"X","Y"]]
    for i in range(data.shape[0]):
        x = data[i][0]
        y = data[i][1]
        ds0 = M * (np.matrix([x,y, 1.0]).T)
        ds1=[i+1,float(ds0[0]),float(ds0[1])]
        data_warp.append(ds1[:3])
    data_warp=np.array(data_warp).reshape(data.shape[0]+1,3)
    return data_warp

def get_TRE(data,data1,M,rows, cols):
    '''
    get the rTRE of data
    params:
        data: coordinates of landmarks in the source image
        data1: coordinates of landmarks in the target image
        M: trasnform information
        rows,cols: the size of image
    return:
        median_rTRE: the median_rTRE of coordinates
        max_rTRE: the max_rTRE of coordinates
        mean_rTRE: the mean_rTRE of coordinates
        save_data: the new coordinates to save
    '''
    data_warp = []
    save_data=[[0,"X","Y"]]
    for i in range(data.shape[0]):
        x = data[i][0]
        y = data[i][1]
        ds0 = M * (np.matrix([x,y, 1.0]).T)
        ds1=[i+1,float(ds0[0]),float(ds0[1])]
        data_warp.append(ds0[:2])
        save_data.append(ds1[:3])
    data_warp=np.array(data_warp).reshape(data.shape)
    a,b=data.shape
    save_data=np.array(save_data).reshape(a+1,b+1)
    TRE = np.linalg.norm((data1 - data_warp), axis=1, ord=2)
    r = (np.sqrt(np.power(rows, 2) + np.power(cols, 2)))
    rTRE = TRE / r
    rTRE = np.sort(rTRE)
    k = int((rTRE.size) / 2)
    median_rTRE = rTRE[k]
    max_rTRE = max(rTRE)
    mean_rTRE = np.mean(rTRE)
    return median_rTRE,max_rTRE,mean_rTRE,save_data

def resizeimage(img,shrink_num):
    '''
    resize image,image BGR to gray
    params:
        img: image array,BGR
        shrink_num: image size scaline factor
    return:
        shrink: resized image,array,BGR
        img1_gray: reiszed image ,array,gray
        size: the new size of image
    '''
    height, width= img.shape[:2]
    size = (int(width * (1/shrink_num)), int(height * (1/shrink_num)))
    shrink = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(shrink, cv2.COLOR_BGR2GRAY)
    return shrink,img_gray,size

def drawpic(dimg1,dimg2):
    '''
    draw a image that is horizontally joined by the two images that input
    params:
        dimg1: image array ,
        dimg2: image array,
    return:
        I: the image that underline the four quadrants, array , 
    '''
    mh1,mw1=int(dimg1.shape[0]/2), int(dimg1.shape[1]/2)
    mh2,mw2=int(dimg2.shape[0]/2), int(dimg2.shape[1]/2)
    I = np.zeros((dimg2.shape[0],dimg1.shape[1]+dimg2.shape[1],3))
    ww=dimg1.shape[1]
    www=dimg1.shape[1]+dimg2.shape[1]
    hh=dimg2.shape[0]
    I[0:dimg1.shape[0],0:dimg1.shape[1],:]=dimg1
    I[0:dimg2.shape[0],dimg1.shape[1]:,:]=dimg2
    cv2.line(I,(0,mh1),(www,mh2),(255,0,0),5)
    cv2.line(I,(mw2+ww,0),(mw2+ww,hh),(255,0,0),5)
    cv2.line(I,(mw1,0),(mw1,hh),(255,0,0),5)
    return I

def Key(region,con=None):
    '''
    bulid class Keypoint by the region
    params :
        region: point array ,shape(-1,2)
        con: contour that points set get by cv2
    return: a class Keypoint that contain center,angle,size ,etc
    '''
    ellipse = cv2.fitEllipse(region)
    center = ellipse[0]
    if con==None:
        ass=1
    else:
        ass=cv2.pointPolygonTest(con,center,True)
    if ass<0:
        return 
    else:
        size = np.sqrt(ellipse[1][0] * ellipse[1][1])*4
        angle = ellipse[2]
        Keypoint=cv2.KeyPoint(x=center[0], y=center[1], _angle=angle,
                                              _size=size, _response=0,
                                              _octave=0, _class_id=-1)
        return Keypoint

def extract_MSER_surf(img1_gray,diag,min=None,max=None):
    '''
    get the MSER features of picture
    params :
        img1_gray: gray image array type uint8
        min: the min area that features could be,int
        max: the max area that features could be,int
    return :
        desc_s: the surf describe vector of MSER features
        kp_s: the coordinates of Keypoints (2,n) type float64
        ks_list: the list of class Keypoint  
    '''
    if min==None:
        min=int(diag*0.06)
    if max==None:
        max=6000
    mser = cv2.MSER_create(_delta=1,_min_area=min,_max_area=max,)
    kpkp_source = mser.detect(img1_gray)
    regions_source, boxes_source = mser.detectRegions(img1_gray)
    kp_source=[Key(p_s) for p_s in regions_source]
    k1=np.array([p.pt for p in kp_source]).T 
    k2=np.array([p.pt for p in kpkp_source]).T
    p_idx0=np.in1d(k1[0],k2[0])
    p_idx1=np.in1d(k1[1],k2[1])
    p_idx=p_idx0*p_idx1
    idx=np.where(p_idx==True)[0]
    ks_list=[kp_source[i] for i in idx]
    surf = cv2.xfeatures2d.SURF_create()
    desc_s= surf.compute(img1_gray, ks_list)[1]
    kp_s = np.array([p.pt for p in ks_list]).T  
    del kpkp_source,kp_source,k1,k2,p_idx0,p_idx1,p_idx
    gc.collect()
    return desc_s,kp_s,ks_list
    
def match_BFMatcher_11(desc_s, desc_t,kp_s,kp_t,thre):
    '''
    find match of Keypoins by the distance of described vector
    params:
        desc_s: surf described vector of source picture 
        desc_t: surf described vector of target picture 
        kp_s: coordinates of Keypoints from source image (2,n) type float64
        kp_t: coordinates of Keypoints from target image (2,n) type float64
        thre: threshold used to filter Euclidean distances between cooridinate pairs(kp_s to kp_t)
    return:
        matches: a list Class Dmatch that made up of pairs of Class keypoints
        newkp_s: sorted by matches that the coordinates of Keypoint from source image, (2,n) type float64
        newkp_t: sorted by matches that the coordinates of Keypoint from target image, (2,n) type float64
        fit_pos: separate index of Keypoints list in array
    '''
    bf = cv2.BFMatcher(normType=cv2.NORM_L1, crossCheck=True)
    matches=[]
    matches1=[]
    newkp_s=np.zeros((0,2))
    newkp_t=np.zeros((0,2))
    for i in range(len(kp_s[0])):
        kps=np.array(kp_s[:,i]).reshape(2,1)
        dis=np.sum((kp_t-kps)**2,0)
        dis=np.sqrt(dis)
        inl=np.where(dis[:]<thre)[0]##if you don't want the thre that input you can use a number here instead thre
        if len(inl)>0:
            ds=np.array(desc_s[i]).reshape(1,-1)
            dt=np.array(desc_t[inl]).reshape(len(inl),-1)
            match_i=bf.match(ds,dt)
            j=str(inl[match_i[0].trainIdx])
            j=int(j)
            match_i[0].queryIdx=i
            match_i[0].trainIdx=j
            matches1.append(match_i[0])
    matches1 =sorted(matches1,key=lambda x:x.distance)
    fit_pos=np.array([[m.queryIdx,m.trainIdx] for m in matches1])
    n = fit_pos.shape[0]
    idx=np.unique(fit_pos[:,1],True)
    if not len(idx[1])==n:
        fit_pos =fit_pos[idx[1],:]
        matches1=[matches1[i] for i in idx[1]]
    newkp_s=kp_s[:,fit_pos[:,0]]
    newkp_t=kp_t[:,fit_pos[:,1]] 
    print("point numbers for RANSAC:"+str(len(matches1)))
    return matches,newkp_s,newkp_t,fit_pos  