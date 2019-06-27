# -*- coding: utf-8 -*-
import cv2
import gc
import numpy as np
import Ransac as ransac
import matplotlib.pyplot as plt
from compute import resizeimage,drawpic,extract_MSER_surf,match_BFMatcher_11
  
class Feature():
    def __init__(self,img1,img2,shrink_num,min=None,max=None,thre=None):
        '''
        resize image and RGBtogray,define callable properties within a class
        params:
            img1: image array,BGR
            img2: image array,BGR
            shrink_num: image size scaline factor (int,float)
        self:
            dimg1: resized image,array,RGB
            dimg2: resized image,array,RGB
            img1_gray: reiszed image ,array,gray
            img2_gray: reiszed image ,array,gray
            shrink_num: image size scaline factor >1 shrink 
            thre: threshold used to filter Euclidean distances between cooridinate pairs
        '''
        shrink1,img1_gray,size1=resizeimage(img1,shrink_num)
        shrink2,img2_gray,size2=resizeimage(img2,shrink_num)
        self.min=min
        self.max=max
        self.img1_gray=img1_gray
        self.img2_gray=img2_gray
        self.dimg1=shrink1
        self.dimg2=shrink2
        self.shrink_num=shrink_num
        self.height2=size2[1]
        self.width2=size2[0]
        ''' thre is defined according to the diagonal length of the image'''
        self.diag=np.sqrt(size2[0]**2+size2[1]**2)
        if thre==None:
            if self.diag>2500:
    #            self.thre=250
                self.thre=int(self.diag*0.08)
            else:
                self.thre=200
        else:
            self.thre=thre

    def match_first(self):
        '''
        use the callable properties defined to get the the MSER features of pictures and match by the surf described
        vector,and draw the picture with matching pairs of two images
        params: 
            defined properties:img1_gray,img2_gray,dimg1,dimg2,thre
        return:
            nkp_s: coordinates of dimg1
            nkp_t: coordinates of dimg2
            Ir: the picture with matching pairs of two images
            lenmatch: string contain match numbers and the feature numbers of img1_gray and img2_gray
        '''
        desc_s, kp_s, ks_list= extract_MSER_surf(self.img1_gray,self.diag,self.min,self.max)
        desc_t, kp_t, kt_list= extract_MSER_surf(self.img2_gray,self.diag,self.min,self.max)
        I_s=self.dimg1.copy()
        I_t=self.dimg2.copy()
        lenmatch=''
        dmatch,nkp_s,nkp_t,fit_pos =match_BFMatcher_11(desc_s, desc_t,kp_s,kp_t,self.thre) 
        draw_params2 = dict(matchColor=(255,0,0), singlePointColor=(255,0,0), # draw matches in blue color
                                       matchesMask=None, 
                                       flags=2)
        I_s=cv2.drawKeypoints(I_s,ks_list,I_s, color=(255, 0, 0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        I_t=cv2.drawKeypoints(I_t,kt_list,I_t, color=(255, 0, 0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        Ir = cv2.drawMatches(I_s, ks_list,I_t, kt_list, dmatch, None, **draw_params2)
        font=cv2.FONT_HERSHEY_SIMPLEX
        ls=str(len(ks_list))
        lt=str(len(kt_list))
        lenmatch='m-'+str(len(nkp_s[0]))+'-s-'+str(ls)+'-t-'+str(lt)
        cv2.putText(Ir,'lenth='+str(lenmatch),(100,200),font,10,(255,0,0),2)
        del desc_s, kp_s, ks_list,desc_t, kp_t, kt_list,dmatch,
        gc.collect()
        return nkp_s,nkp_t,Ir,lenmatch
        
    def match_sec(self,nkp_s,nkp_t):
        '''
        use coordinates of images and RANSAC to find the interior point,and estimation affine martix
        params:
            nkp_s: coordinates of dimg1
            nkp_t: coordinates of dimg2
        return:
            M: affine martix,array, shape(2,3)
            in_s: coordinates of the interior points of dimg1
            in_t: coordinates of the interior points of dimg2
            threshold: the threshold that Ransac used
        '''
        nkp1_s=nkp_s
        nkp1_t=nkp_t
        M, inliers,threshold = ransac.affine_matrix(nkp1_s, nkp1_t)
        in_s=nkp1_s[:,inliers[0]]
        in_t=nkp1_t[:,inliers[0]]
        if M is not None:
            if len(in_s[0])<=15:
                M=cv2.getRotationMatrix2D((0,0),0,1)
        return M,in_s,in_t,threshold
    
    def draw_register(self,I,ww,in_s,in_t):
        '''
        draw the pairs of interior points in a mosaic image
        params:
            I: mosaic image
            ww: the width of img1_gray, int
            in_s: coordinates of the interior points of dimg1 (2,n),float
            in_t: coordinates of the interior points of dimg2 (2,n),float
        return:
            I: the image has drawn the match by the coordinates of the interior points
        '''
        for p in range(len(in_s[0])):
            p1=in_s[:,p]
            p2=in_t[:,p]
            cv2.line(I,(int(p1[0]),int(p1[1])),(int(p2[0])+ww,int(p2[1])),(0,0,255),3)
            cv2.circle(I,(int(p1[0]),int(p1[1])),10,(0,0,255),3)
            cv2.circle(I,(int(p2[0]+ww),int(p2[1])),10,(0,0,255),3)
        return I
    
    def register(self):
        '''
        use the defined properties and functions to achieved the feature point registration
        params:
            the defined properties and functions
        return:
            M: affine martix,array, shape(2,3)
            I: the image has drawn the match of interior points
            lenmatch: string contain match numbers and the feature numbers
            threshold: the threshold that Ransac used
            interiornum: the number of interior points
            Ir: the image has drawn all the match of feature points
        '''
        nkp_s,nkp_t,Ir,lenmatch=self.match_first()
        M,in_s,in_t,threshold=self.match_sec(nkp_s,nkp_t)
        I=drawpic(self.dimg1,self.dimg2)
        ww=self.dimg1.shape[1]
        I=self.draw_register(I,ww,in_s,in_t)
        interiornum=len(in_s[0])
        return M,I,lenmatch,threshold,interiornum,Ir
        