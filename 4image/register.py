# -*- coding: utf-8 -*-
import math
import cv2
from scipy import signal
from xml_draw import resizeimage
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt
class Pre():
    def __init__(self,img1,img2,size):
        '''
        resize image and define callable properties within a class
        params:
            img1: image array,BGR
            img2: image array,BGR
            size: the max image resolution that resized
        self:
            dimg1: resized image,array,RGB
            dimg2: resized image,array,RGB
            img1_gray: reiszed image ,array,gray
            img2_gray: reiszed image ,array,gray
            shrink_num: image size scaline factor >1 shrink 
            height2: the height of the img2
            width2: the width of the img2
        '''
        height2, width2= img2.shape[:2]
        height1, width1= img1.shape[:2]
        shrink_num=math.ceil(max(height2, width2,height1, width1)/size)
        shrink1,img1_gray,size1=resizeimage(img1,shrink_num)
        shrink2,img2_gray,size2=resizeimage(img2,shrink_num)
        self.dimg1=shrink1
        self.dimg2=shrink2
        self.img1_gray=img1_gray
        self.img2_gray=img2_gray
        self.shrink_num=shrink_num
        self.height2=height2
        self.width2=width2

    def crop_cross(self):
        '''
        crop the black cross in the image
        self:img1_gray,img2_gray
        return:
            dst1: image croped the black cross ,array,gray
            dst2: image croped the black cross ,array,gray
        '''
        dst1=self.img1_gray
        dst2=self.img2_gray
        a1=np.mean(dst1[-10:,0:10])
        a2=np.mean(dst2[-10:,0:10])
        dst1=np.array((a1-dst1)*(dst1[:]<50)+dst1).astype('uint8')
        dst2=np.array((a2-dst2)*(dst2[:]<50)+dst2).astype('uint8')
# =============================================================================
#         for i in range(1,dst1.shape[0]):
#             for j in range(1,dst1.shape[1]):
#                 if dst1[i,j]<50:
#                     dst1[i,j]=a1
#         for i in range(1,dst2.shape[0]):
#             for j in range(1,dst2.shape[1]):
#                 if dst2[i,j]<50:
#                     dst2[i,j]=a2
# =============================================================================
        if max(a1,a2)>200:
            dst1=255-dst1
            dst2=255-dst2
        return dst1,dst2
    
    def get_center(self,img):
        '''
        find the centroid of img,and threshold the img
        params:
            img: array,gray image
        return:
            _x: image centroid coordinates in the X direction
            _y: image centroid coordinates in the Y direction
            I: image after threshold'''
        m10=m00=m01=0
        thre=filters.threshold_li(img)
        mask=img[:]>thre
        I=mask*img
        plt.imshow(I)
        plt.show()
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                m10+=(1*j*(I[i][j]))
                m01+=(i*1*(I[i][j]))
                m00+=(I[i][j])
        _x=(m10/m00)
        _y=(m01/m00)
        return _x,_y,I
    
    def gradient(self,image):
        '''
        use prewitt compute the normalized gradient of image
        params: 
            image: gray image
        return:
            img1_gradient: normalized gradient of image,np.vstack(x,y),shape(2*image.shape)
        '''
        prewitt_x=np.array(([-1,0,1],[-1,0,1],[-1,0,1]))
        prewitt_y=np.array(([-1,-1,-1],[0,0,0],[1,1,1]))
    #    sobel_x=np.array(([-1,0,1],[-2,0,2],[-1,0,1]))
    #    sobel_y=np.array(([-1,-2,-1],[0,0,0],[1,2,1]))
        weight_x,weight_y=prewitt_x,prewitt_y
        imgto1_x=np.abs(signal.convolve2d(image,weight_x,boundary='symm',mode='same'))
        imgto1_y=np.abs(signal.convolve2d(image,weight_y,boundary='symm',mode='same'))
        x1=np.array(imgto1_x,dtype=np.float64)
        x11=np.array(imgto1_y,dtype=np.float64)
        dx1=np.sqrt(x1**2+x11**2+0.01)
        img1_gradient=np.vstack((x1/dx1,x11/dx1))
        return img1_gradient
    
    def NGF(self,I1,I2):
        '''compute the NGF of the image input
        params:
            I1:gray image
            I2:gray image
        return:
            distance: a value represents the image gradient difference
            ngf: a array represent the gradient difference of corresponding position
        '''
        img1_gradient=self.gradient(I1)  
        img2_gradient=self.gradient(I2)            
        ngf=1-(img1_gradient*img2_gradient)**2
        distance=np.sum(ngf)
        return distance,ngf
    
    def pre_register(self):
        '''
        find the rotation and the translation of two image,achieve pre-register
        params:
            self: defined callable properties,and defined functions
        return:
            M1: the transformation martix that find which contain the rotation and translation
        '''
        dst1,dst2=self.crop_cross()
        p1,q1,dst1_thre=self.get_center(dst1)
        p2,q2,dst2_thre=self.get_center(dst2)
        gdst2=self.gradient(dst2_thre)
        dis=[]
        dis_ssd=[]
        for ang in range(32): 
                ###   image translation by the vectors that from the center of moving image 
                ###   to the center of fixed image
            tranform=np.float64([[1,0,(p2-p1)],[0,1,(q2-q1)]])
            warp1=cv2.warpAffine(dst1_thre,tranform,(dst2.shape[1],dst2.shape[0]))
                ###   image rotation from 0 to 360 
            angle=(360/32)*ang
            rotate=cv2.getRotationMatrix2D((int(p2),int(q2)),angle,1)
#            rotate[0,2]=rotate[0,2]+(p2-p1)
#            rotate[1,2]=rotate[1,2]+(q2-q1)
            I1=cv2.warpAffine(warp1,rotate,(dst2.shape[1],dst2.shape[0]))
            I2=dst2_thre
             ###   compute NGF
            distance,ngf=self.NGF(I1,I2)
            ss=self.gradient(I1)-gdst2
            ssd=np.sum(ss**2)/(gdst2.shape[0]*gdst2.shape[1])
            dis_ssd.append(ssd)
            dis.append(distance)
        dis_norm=(dis-np.min(dis))/(np.max(dis)-np.min(dis))
        dissort=sorted(dis_norm)
        k0=np.argmin(dis)
        k1=np.where(dis_norm==dissort[1])[0][0]
        d_dis=dissort[1]-dissort[0]
        if 0.01<d_dis<0.02:
            if dis_ssd[k0]>dis_ssd[k1]:
                k=k1
            else:
                k=k0
        else:
            k=k0
        angle=(360/32)*k
        M1=cv2.getRotationMatrix2D((p1*self.shrink_num,q1*self.shrink_num),angle,1)
        M1[0,2]=M1[0,2]+(p2-p1)*self.shrink_num
        M1[1,2]=M1[1,2]+(q2-q1)*self.shrink_num
        return M1