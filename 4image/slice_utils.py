# -*- coding: utf-8 -*-
import numpy as np
import os
import cv2

def cut_moving_region(region,slide1,length,save_path,base,gidx,withcontour=None,overlap=None):
    '''
    cut out the fixed size picture,and save(the size of area is bigger than the right size)
    params:
        region: the area required,a coordinate list,(-1,2)
        slide1: image that open by opsl
        length: the fixed size
        save_path: the path to save
        base: the basename of image
        gidx: the index of region of image
        withcontour: whether draw contours in the picture or not
        overlap: whether allow overlap when cutting images
    return:
        position: a list of the coordinate in the top left corner
    '''
    cnt=np.array(region).reshape(-1,1,2).astype('int32')
    x,y,w,h=cv2.boundingRect(cnt)
    imglist=[]
    contours=[]
    saveimglist=[]
    position=[]
    if overlap ==True:
        new_w=max(w,length)
        new_h=max(h,length)
        xx=int(x+(w-new_w)/2)
        yy=int(y+(h-new_h)/2)
        slicex,stepx=divmod(w,length)
        slicey,stepy=divmod(h,length)
        img=slide1.read_region((xx,yy),0,(new_w,new_h))
        img2=np.asarray(img)
        if stepx>0:
            slicex=slicex+1
        if stepy>0:
            slicey=slicey+1
        for ix in range(slicex):
            for iy in range(slicey):
                inx=ix*2+iy
                save_name=str(base)[:-5]+'-square'+str(length)+'-'+str(gidx)+'-'+str(inx)+'.jpg'
                saveimg=os.path.join(save_path,save_name)
                if ix>0:
                    sx=stepx+length*(ix-1)
                else:
                    sx=0
                if iy>0:
                    sy=stepy+length*(iy-1)
                else:
                    sy=0
                img3=img2[sy:sy+length,sx:sx+length]
                cntlist=[]
                pt=[xx+sx,yy+sy]
                cntlist.append(cnt-np.array(pt).reshape(1,1,2).astype('int32'))
                saveimglist.append(saveimg)
                imglist.append(img3)
                contours.append(cntlist)
                position.append(pt)
    else: 
        slicex,stepx=divmod(w,length)
        slicey,stepy=divmod(h,length)
        new_w=slicex*length
        new_h=slicey*length
        xx=int(x+(w-new_w)/2)
        yy=int(y+(h-new_h)/2)
        img=slide1.read_region((xx,yy),0,(new_w,new_h))
        img2=np.asarray(img)
        for ix in range(slicex):
            for iy in range(slicey):
                inx=ix*2+iy
                save_name=str(base)[:-5]+'-square'+str(length)+'-'+str(gidx)+'-'+str(inx)+'.jpg'
                saveimg=os.path.join(save_path,save_name)
                saveimglist.append(saveimg)
                img3=img2[iy*length:(iy+1)*length,ix*length:(ix+1)*length]
                imglist.append(img3)
                cntlist=[]
                pt=[xx+ix*length,yy+iy*length]
                cntlist.append(cnt-np.array(pt).reshape(1,1,2).astype('int32'))
                contours.append(cntlist)
                position.append(pt)
    for i in range(len(saveimglist)):
        img3=imglist[i]
        cntlist=contours[i]
        saveimg=saveimglist[i]
        if withcontour is True:
            imgcopy=cv2.drawContours(img3,cntlist,-1,(255,255,0),2)
            imgcopy=cv2.cvtColor(imgcopy,cv2.COLOR_BGRA2RGB)
            cv2.imwrite(saveimg,imgcopy)
        else:
            img3=cv2.cvtColor(img3,cv2.COLOR_BGRA2RGB)
            cv2.imwrite(saveimg,img3)
    return position

def cutfixedregion(region,slide1,length,save_path,base,gidx,withcontour=None):
    '''
    cut out the right size picture,and save(the size of area is smaller than the right size or similar)
    params:
        region: the area required,a coordinate list,(-1,2)
        slide1: image that open by opsl
        length: the fixed size
        save_path: the path to save
        base: the basename of image
        gidx: the index of region of image
        withcontour: whether draw contours in the picture or not
    return:
        pt: the coordinate in the top left corner
    '''
    cnt=np.array(region).reshape(-1,1,2).astype('int32')
    x,y,w,h=cv2.boundingRect(cnt)
    new_w=length
    new_h=length
    save_name=str(base)[:-5]+'-square'+str(length)+'-'+str(gidx)+'.jpg'
    saveimg=os.path.join(save_path,save_name)
    xx=int(x+(w-new_w)/2)
    yy=int(y+(h-new_h)/2)
    img=slide1.read_region((xx,yy),0,(new_w,new_h))
    pt=[xx,yy]
    img2=np.asarray(img)
    if withcontour is True:
        cntlist=[]
        cntlist.append(cnt-np.array(pt).reshape(1,1,2).astype('int32'))
        imgcopy=img2.copy()
        imgcopy=cv2.drawContours(imgcopy,cntlist,-1,(255,255,0),2)
        imgcopy=cv2.cvtColor(imgcopy,cv2.COLOR_BGRA2RGB)
        cv2.imwrite(saveimg,imgcopy)
    else:
        img3=cv2.cvtColor(img2,cv2.COLOR_BGRA2RGB)
        cv2.imwrite(saveimg,img3)
    return pt 

def cutregion(region,slide1,length,save_path,base,gidx,withcontour=None,write=True):
    '''
    cut the fixed size image according to the provided area 
    params:
        region: the area required,a coordinate list,(-1,2)
        slide1: image that open by opsl
        length: the fixed size
        save_path: the path to save
        base: the basename of image
        gidx: the index of region of image
        withcontour: whether draw contours in the picture or not
        write: whether save image or not
    return:
       position: a list of the coordinate in the top left corner
       image_num: the number of the fixed size images  
    '''
    cnt=np.array(region).reshape(-1,1,2).astype('int32')
    x,y,w,h=cv2.boundingRect(cnt)
    position=[]
    image_num=0
    if w>2*length and h>2*length:
         position1=cut_moving_region(region,slide1,length,save_path,base,gidx,withcontour)
         image_num+=len(position1)
    else:
        position.append(cutfixedregion(region,slide1,length,save_path,base,gidx,withcontour))
        image_num+=1
    for pt in position1:
        position.append(pt)
    return position,image_num

def cutrectangle(region,slide1,save_path,base,gidx,withcontour=None,write=True):
    '''
    cut the rectangle image according to the provided area 
    params:
        region: the area required,a coordinate list,(-1,2)
        slide1: image that open by opsl
        save_path: the path to save
        base: the basename of image
        gidx: the index of region of image
        withcontour: whether draw contours in the picture or not
        write: whether save image or not
    return: None 
    '''
    cnt=np.array(region).reshape(-1,1,2).astype('int32')
    x,y,w,h=cv2.boundingRect(cnt)
    img=slide1.read_region((x,y),0,(w,h))
    img2=np.asarray(img)
    if write==True:
        save_name=str(base)[:-5]+'-rectangle-'+str(gidx)+'.jpg'
        saveimg=os.path.join(save_path,save_name)
        if withcontour is True:
            cntlist=[]
            cntlist.append(cnt-np.array([x,y]).reshape(1,1,2).astype('int32'))
            imgcopy=img2.copy()
            imgcopy=cv2.drawContours(imgcopy,cntlist,-1,(255,255,0),2)
            imgcopy=cv2.cvtColor(imgcopy,cv2.COLOR_BGRA2RGB)
            cv2.imwrite(saveimg,imgcopy)
        else:
            img3=cv2.cvtColor(img2,cv2.COLOR_BGRA2RGB)
            cv2.imwrite(saveimg,img3)
    return

def getposition_area(regions):
    '''
    get the coordinate in the upper left corner of the regionbox,and the size of regionbox 
    params:
        regions:list of region
    return: 
        position:the coordinate in the upper left corner of the regionbox(x,y),and the size of regionbox(w,h)
    '''
    position=[]
    for region in regions:
        cnt=np.array(region).reshape(-1,1,2).astype('int32')
        x,y,w,h=cv2.boundingRect(cnt)
        position.append([x,y,w,h])
    return position

def cutrandomimage(mask,downsample,length,slide1,base,save_path,image_num):
    '''
    cut the fixed image randomly from the mask provided,and save images according to the path
    params:
        mask: the area to cut the fixed image
        downsample: minification that the mask is
        length: the fixed size
        slide1: image that open by opsl
        base: the basename of image
        save_path: the path to save
        image_num: the number of the fixed image 
    return :None
    '''
    mask3=np.zeros(mask.shape,np.uint8)
    position_array=(np.array(np.where(mask==255))*downsample).astype('int64')
    slength=int(length/downsample)+1
#    unique=np.unique(position_array,axis=1)
    img_num=0
    while img_num<image_num:
        ridx=np.random.sample(range(0,len(position_array[0])),1)
        randompt=position_array[:,ridx]
        newx=int(randompt[1])
        newy=int(randompt[0])
#        print(newx,newy)
        randomimage=slide1.read_region((newx,newy),0,(length,length))
        randomimage=np.asarray(randomimage)
        gray=cv2.cvtColor(randomimage,cv2.COLOR_BGRA2GRAY)
        white=np.sum(gray[:]>200)/(length*length)
        black=np.sum(gray[:]<50)/(length*length)
        if white>0.14:
            continue
        if black>0.1:
            continue
        _x=int(newx/downsample)
        _y=int(newy/downsample)
        cv2.rectangle(mask3,(_x,_y),(_x+slength,_y+slength),127,-1) 
        save_name=str(base)[:-5]+'-random'+str(length)+'-'+str(img_num)+'.jpg'
        saverandomimg=os.path.join(save_path,save_name)
#        font=cv2.FONT_HERSHEY_SIMPLEX
#        cv2.putText(randomimage,'judge='+str(white)[:4]+'-'+str(black)[:4],(20,30),font,1,(255,0,0),1)
        randomimage=cv2.cvtColor(randomimage,cv2.COLOR_BGRA2RGB)
        cv2.imwrite(saverandomimg,randomimage)
        img_num+=1
    ra=mask-mask3
    save_name=str(base)[:-5]+'-mask'+str(length)+'.jpg'
    saveimg=os.path.join(save_path,save_name)
    cv2.imwrite(saveimg,ra)
    return





# =============================================================================
# def cut_moving_region_with_overlap(x,y,w,h,slide1,length,save_path,base,gidx,withcontour=None):
#     cnt=np.array(region).reshape(-1,1,2).astype('int32')
#     x,y,w,h=cv2.boundingRect(cnt)
#     new_w=max(w,length)
#     new_h=max(h,length)
#     xx=int(x+(w-new_w)/2)
#     yy=int(y+(h-new_h)/2)
#     slicex,stepx=divmod(w,length)
#     slicey,stepy=divmod(h,length)
#     if stepx>0:
#         slicex=slicex+1
#     if stepy>0:
#         slicey=slicey+1
#     img=slide1.read_region((xx,yy),0,(new_w,new_h))
#     img2=np.asarray(img)
#     for ix in range(slicex):
#         for iy in range(slicey):
#             inx=ix*2+iy
#             save_name=str(base)[:-5]+'-square'+str(length)+'-'+str(gidx)+'-'+str(inx)+'.jpg'
#             saveimg=os.path.join(save_path,save_name)
#             if ix>0:
#                 sx=stepx+length*(ix-1)
#             else:
#                 sx=0
#             if iy>0:
#                 sy=stepy+length*(iy-1)
#             else:
#                 sy=0
#             img3=img2[sy:sy+length,sx:sx+length]
#             if withcontour is True:
#                 cntlist=[]
#                 cntlist.append(cnt-np.array([xx+sx,yy+sy]).reshape(1,1,2).astype('int32'))
#                 imgcopy=img3.copy()
#                 imgcopy=cv2.drawContours(imgcopy,cntlist,-1,(255,255,0),2)
#                 imgcopy=cv2.cvtColor(imgcopy,cv2.COLOR_BGRA2RGB)
#                 cv2.imwrite(saveimg,imgcopy)
#             else:
#                 img3=cv2.cvtColor(img3,cv2.COLOR_BGRA2RGB)
#                 cv2.imwrite(saveimg,img3)
#     return 
# =============================================================================
