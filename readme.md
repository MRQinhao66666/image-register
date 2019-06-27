#文件夹及文件简单介绍  
##4image  
对自己的四张图像进行配准，这里是用连通域配准  
###文件内容   
**cut-slice.py--对图像感兴趣区域切片的主程序**   
**main.py--图像配准主函数**  
**slice_utils.py--定义一些用于切图的函数**  
**xml_draw.py--定义一些读取标注和使用标注的函数**  
compute.py--定义一些用于图像配准的函数   
feature.py--定义一个特征点配准的类，包含一些特征点配准的函数   
Ransac.py--定义一个包含RANSAC算法使用的函数的类    
register.py--定义一个预配准的类，包括过程和一些使用的函数  
###文件夹内容  
image--图像  (xx.ndpi)   
label--标注  (xx.ndpi.ndpa)  

#文件注释-4image   

##main.py   

	......   
	//获取图像的标注（区域和针）以及区域对应的颜色列表  
	regions1,pins1,colorlist=xml.get_region_pin(label_name,slide1,dims)  
	//在对应维度的图像上按照颜色列表画标注  
	imglabel=xml.draw(slide_thumbnail1,regions1,pins1,colorlist)  
	//获取图像连通域mask集合（二值图像0，255）以及对应的轮廓坐标集合  
	mask1s,contours=xml.get_mask(img1,'slice')  
	mask2s,contour2s=xml.get_mask(img2,'slice')  
	//根据图像连通域的轮廓，判断图像的标注所在的连通域，返回索引  
	idx_region,idx_pin=xml.judge_label(regions1,pins1,contours)  
	//对于每个在模板集合中的模板，获得模板对应的图像  
	for m  in range(len(mask1s)):  
		mask1=mask1s[m][:]>0  
		img1_i=np.zeros(img1.shape,dtype=np.uint8)  
		img2_i=np.zeros(img2.shape,dtype=np.uint8)  
		mask2=mask2s[m][:]>0  
	for i in range(3):  
		img1_i[:,:,i]=img1[:,:,i]*mask1  #+(1-mask1)*255  black/white
		img2_i[:,:,i]=img2[:,:,i]*mask2  #+(1-mask2)*255  black/white
	//利用获得的对应图像进行预配准，特征点配准等操作，返回变换矩阵MI  
	......  
	//将各个连通域获得的变换矩阵放入到一个集合中，扭曲后的图像放入一张图显示，内点匹配图放入一张图显示  
	M_lis.append(MI)  
	img3=img3+img3_i*(img3_i[:]<255)  
	I=I+I_  
	//获取新标注的坐标（区域和针）并且在图像上显示  
	regions2,pins2=xml.get_newlabel_num(regions1,pins1,idx_region,idx_pin,M_lis)  
	img3_im=Image.fromarray(np.uint8(img3))  
	img2_im=Image.fromarray(np.uint8(img2))  
	img3_label=xml.draw(img3_im,regions2,pins2,colorlist)  
	img2_label=xml.draw(img2_im,regions2,pins2,colorlist)  

##xml_draw.py   

	get_region_pin(labelname,slide,dims)  
	：读取标注文件，获得标注区域的位置（freehand--自由区域，pin--针/点） 
	文件中的x，y中记录的是绝对位置，以纳米作为单位的物理位置，需要将这些位置转换为在图像中的相对位置，因此需要读取文件的mpp-x,mpp-y
	(每个方向上单位相对位置代表的微米长度），以及(XOffsetFromSlideCentre，YOffsetFromSlideCentre)物理中心也就是图像中心（h/2，w/2)对应的物理位置，图像尺寸（h,w),  
	最后通过一个变换公式获得相对位置：xx=((x-XOffsetFromSlideCentre)/(mppx*1000))+h/2，yy=((y-YOffsetFromSlideCentre)/(mppy*1000))+w/2  
	然后根据输入的维度对应的缩小比例对获得的相对位置进行缩小，从而得到在不同尺度下的相对坐标  
	  
	get_newlabel(regions,pinlist,M)  
	：输入原始的标注的位置和一个变换矩阵，获得标注经过变换后的新的位置  
	  
	draw(im,regions,pinlist,colorlist)  
	：输入Image格式图像，在图上画不同颜色的区域和固定颜色的指针的标注  
	  
	get_mask(image,mod)  
	：针对四张病理图像获取其中非背景的区域的mask，mod为slice返回一个由连通域mask组成的列表和轮廓坐标矩阵，mod为total返回整张图的mask和轮廓坐标矩阵的集合  
	
	getmainmask(masklist,idx_region)  
	：输入表明所有标注区域所在的轮廓的索引，以及连通域mask列表集合，找出包含标注最多的mask返回  
	
	mask_reduce_region(mask1,position,length,downsample)  
	：输入一个mask，这个mask代表所有的有颜色的区域，position记录所有包含标注的矩形区域的左上角坐标和矩形的长和宽（坐标的尺度对应于mask对应的尺度），在这个mask上减去标注对应的矩形和可能会裁剪到标注矩形的相邻patch（左上，上，左，三个方向的patch减去）
	返回一个新的mask，这个mask中大于0的区域对应的原图像切patch绝对不会出现切到包含标注的情况  
	
	judge_label(regions,pins,contours,returnnew=None)  
	：判断标注属于那个轮廓返回区域的轮廓索引和指针的轮廓索引，如果returnnew为True,还要返回新的根据轮廓重新分布的标注，就是一个和轮廓数量一致的列表，列表中放着对应轮廓的标注  
	
	get_newlabel_num(regions,pins,idx_region,idx_pin,M_lis)  
	：根据标注及标注的轮廓索引和轮廓对应的变换矩阵，生成新的标注的位置  

##cut-slice.py  

	//定义一些参数  
	withcontour=None     是否在图像上画轮廓   
	length=300           图像切patch的尺寸   
	dim=4                mask的维度  4级   
	num=50               需要随机切的图像的数量   
	......  
	//获取4级下的图像  
	slide_thumbnail1 = slide1.get_thumbnail(slide1.level_dimensions[dim])  
	image=np.array(slide_thumbnail1)  
	//获取4级图像的连通域mask列表和轮廓坐标列表  
	masklist,cnts=xml.get_mask(image,'slice')  
	//获取4级图像对应的标注位置和颜色列表  
	regions1,pins1,colorlist=xml.get_region_pin(label_name,slide1,dim)  
	//获取所有标注区域的外接矩形左上角坐标和长宽  
	position_all=slice_utils.getposition_area(regions1)  
	//取标注最多的连通域mask  
	idx_region,idx_pin=xml.judge_label(regions1,pins1,cnts)  
	mask1=xml.getmainmask(masklist,idx_region)  
	//取出图像中绿色和黄色区域的标注的索引  
	greenlist=[]  
	yellowlist=[]  
	for index,nums in enumerate(colorlist):  
		if nums=='#00ff00':  
			greenlist.append(index)  
		if nums=='#ffff00':  
			yellowlist.append(index)  
	//获取0级图像下标注位置  
	regions,pins,colorlist=xml.get_region_pin(label_name,slide1,0)  
	//根据绿色标注的索引从0级标注中取标注坐标，并根据标注区域切patch  
	for gidx in greenlist:  
		print('green')  
		save_path=os.path.join(result_path,'green')  
		if not os.path.exists(save_path):  
			os.makedirs(save_path)  
		region=regions[gidx]  
		#//根据外接矩形的坐标和大小切图像   
		#slice_utils.cutrectangle(region,slide1,save_path,base,gidx,withcontour)  
		position,greenimage_num=slice_utils.cutregion(region,slide1,length,save_path,base,gidx,withcontour)
	//根据黄色标注的索引从0级标注中取标注坐标，并根据标注区域的外接矩形切图  
	for yidx in yellowlist:  
		print('yellow')  
		save_path1=os.path.join(result_path,'yellow')  
		if not os.path.exists(save_path1):  
			os.makedirs(save_path1)  
		region=regions[yidx]  
		slice_utils.cutrectangle(region,slide1,save_path1,base,yidx,withcontour)  
	//获取不包含标注区域的mask（去除左上，上，左三个方向的patch）  
	mask=xml.mask_reduce_region(mask1,position_all,length,downsample)  
	//在mask范围内随机切图  
	save_path2=os.path.join(result_path,'random'+str(num))  
	if not os.path.exists(save_path2):  
		os.makedirs(save_path2)  
	mask3=np.zeros(mask.shape,np.uint8)  
	slice.utils.cutrandomimage(mask,downsample,length,slide1,base,save_path,num)  

##slice_utils.py   

	cut_moving_region(region,slide1,length,save_path,base,gidx,withcontour=None,overlap=None)  
	：对输入的0级标注区域的坐标在0级图像中切出固定大小的patch，patch大小--length，  
	  这个是当输入区域的外接矩形长度比要切的patch大（大于patch长度的两倍）的情况下使用的切图函数，  
	  withcontour控制是否在图像上画轮廓，overlap控制是否允许切图时重叠（重叠则保留外接矩形的大小，在外接矩形上滑动切patch，
	  不重叠则是切一个长宽为patch大小的固定倍数的矩形（标注居中），在矩形内切patch）  

	cutfixedregion(region,slide1,length,save_path,base,gidx,withcontour=None)  
	：对输入的0级标注区域的坐标在0级图像中切出固定大小的patch，这个是输入区域的外接矩形小于或者和patch尺寸差别不大的情况下使用的切图函数，标注居中  

	cutregion(region,slide1,length,save_path,base,gidx,withcontour=None,write=True)  
	：一个封装好的切图函数，会根据标注区域的外接矩形的大小自动使用对应的切图函数（cut_moving_region，cutfixedregion）  

	cutrectangle(region,slide1,save_path,base,gidx,withcontour=None,write=True)  
	：对输入的0级标注区域的坐标在0级图像中切出对应的外接矩形  

	getposition_area(regions)  
	：获取所有标注区域的外接矩形的左上角坐标和矩形的长宽  

	cutrandomimage(mask,downsample,length,slide1,base,save_path,image_num)
	：根据输入的mask获得对应的0级图像，在对应的0级图上随机的切固定大小的patch，只保留切出的图像白色区域小于0.14黑色区域小于0.1的patch切图数量由image_num控制  
所有的切图函数都需要输入一个路径和名字用于保存图像，切图函数不返回图像，直接保存  
  
******  
******    



##train  
利用挑战赛的数据进行配准,没有切连通域  
###文件内容  
**json.py--由挑战赛提供的一个生成名为computer-performances.json的文件，这个文件用于评估电脑的性能，后续提交到挑战赛中的时候挑战赛组织方会利用这个文件将不同电脑上的结果模拟到同一台电脑上运行的情况，比较图像配准的速度**    
**pre-surf.py--使用所有图像根据提供的csv文件实现批量图像配准的主函数**     
**rTRE.py--计算移动图像和固定图像的初始rTRE值**  
**train.py--使用training文件夹的数据实现批量图像配准的主函数**    
compute.py--定义一些用于图像配准的函数     
feature.py--定义一个特征点配准的类，包含一些特征点配准的函数   
Ransac.py--定义一个包含RANSAC算法使用的函数的类   
register.py--定义一个预配准的类，包括过程和一些使用的函数  
### 文件夹内容
all--包括所有图像的image和label  （image，label,dataset_medium.csv)  
training--使用挑战赛中的train数据集  (image,label)  


#文件注释-train  

##josn.py    
这个文件是挑战赛提供的不需要修改，直接运行就可以了。  
（我运行这个文件曾经报错它要求的numpy版本和我使用的版本不一致，我改了版本就可以正常运行，numpy 1.16.3)   

##pre-surf.py    

	//定义图像放缩系数shrink_num和预配准时图像采样的最大尺寸fixedsize  
	shrink_num2 = 5  
	fixedsize = 200  
	string = []  
	......  
	//读取图像和标注  
	data_source = (np.loadtxt(label_source_path, dtype=np.str, delimiter=","))[1:, 1:].astype(np.float)  
	img_source = cv2.imread(file_source_path,1)  
	img_target = cv2.imread(file_target_path,1)  
	//预配准，图像采样至一定大小   
	pre = register.Pre(img_source,img_target,fixedsize)  
	M_warp是预配准过程获得的变换矩阵   
	M_warp = pre.pre_register()    
	width2 = pre.width2  
	height2 = pre.height2  
	img_warp = cv2.warpAffine(img_source, M_warp, (width2, height2))  
	//特征点配准，图像压缩一定程度  
	feature1 = feature.Feature(img_warp,img_target,shrink_num2)  
	【注】：M2是特征点配准返回的变换矩阵，I是待配准图像和参考图像上内点配对的图， 
		lenmatch是一串包含匹配用的特征数，待配准图像和参考图像上找到的特征数的字符串 
		threshold 是RANSAC算法中判断点为内点的阈值（依据） 
		in_num是RANSAC找到的内点数   
		Ir是所有匹配上的特征点配对的图像  
	M2,I,lenmatch,threshold,in_num,Ir = feature1.register()  
	//获取对原图的完整的变换矩阵  
	M = M2
	M[0,2] = 5*M2[0,2]  
	M[1,2] = 5*M2[1,2]  
	M_warp = np.vstack((M_warp,[0,0,1]))  
	M2_ = np.vstack((M,[0,0,1]))  
	M3 = np.dot(M2_,M_warp)  
	......  
##rTRE.py    
	返回一个labelrTRE.txt文件，按列分别表示文件名，中位rTRE，最大rTRE，平均rTRE  
	rTRE指图像中所有标注点对的归一化欧式距离的集合，中位rTRE，最大rTRE，平均rTRE分别取中位数，最大值，平均值  

##train.py   

	获取路径的方式和pre-surf.py中不一样，  
	pre-surf中是根据存放在csv文件中的字符串获得读取路径，读取文件，  
	train是首先一步步循环获得文件夹中的文件所对应的路径，进而读取文件。   
	其他内容一致   
	增加的内容  
	//计算两个图像中点对的rTRE值，以及新的标注的位置，保存   
	rTRE,_,_,data_save= c.get_TRE(data_source, data_target, M3,height2,width2)
	np.savetxt(os.path.join(save_data_path,str(idx)[:-4]+'.csv'),data_save,fmt=('%s,%s,%s'),delimiter=',')   
	//当计算的rTRE过大，将其放入error文件夹  
	if rTRE>0.02:  
		cv2.imwrite(str(error_image), merge)  

#关键内容,两个文件夹都包含的函数文件   
##compute.py   
###fun:  

	get_data(data,M)  
	：根据变换矩阵M和待配准图像中标签的位置data，获得经过变换后的新的标签的位置data_warp  

	get_TRE(data,data1,M,rows, cols)  
	：根据变换矩阵M和待配准图像中标签的位置data，获得经过变换后的新的标签的位置data_warp  
	计算data_warp和参考图像中标签的位置data1之间的欧式距离，根据原始图像的大小（rows, cols）进行归一化  
	获得median_rTRE，max_rTRE，mean_rTRE，  
	另外给data_warp增加行号和第一行获得save_data  

	resizeimage(img,shrink_num)  
	：将图像根据输入的缩小系数进行缩小，同时返回缩小后的图像和对应的灰度图及图像的长和宽  

	drawpic(dimg1,dimg2)  
	：根据输入的两个彩图画一个两张彩图的水平拼接的大图  

	Key(region,con=None)  
	：对输入的区域椭圆拟合，计算区域的中心，大小和方向，构建一个类别为Keypoint的变量  
	当输入的con为None时默认对所有的区域构建Keypoint，如果输入为一个轮廓点集的话，则当输入的区域在轮廓内才构建一个Keypoint变量 
 
**输入的最大最小当效果不好的时候可以自行设置（四张病理图像使用的是100-6000**    
 
	extract_MSER_surf(img1_gray,diag,min=None,max=None)  
	：对输入的灰度图像寻找MSER特征点，并计算特征点的SURF特征描述子，当输入的min和max为None时 默认为min是输入的变量diag的0.06倍,
	max为6000,min,max为函数cv2.MSER_create的参数，控制计算MSER特征（大于min,小于max的范围内）
	返回:desc_s--特征描述向量的集合，kp_s--特征点中心坐标的集合，ks_list--Keypoint变量集合  

**距离每个特征中心的欧式距离可以自行设置以获得更好的效果**   

	match_BFMatcher_11(desc_s, desc_t,kp_s,kp_t,thre)  
	：首先获得离待配准图像中特征的中心的欧式距离在一定范围内的集合，
	在这个集合中利用bfmatch寻找对应的描述向量（desc_s,desc_t)最小的欧式距离并匹配（Dmatch)，
	根据这些匹配在对应keypoint变量集合（desc_t)中的索引替代在范围集合(dt)内的索引，
	然后，将这些匹配（Dmatch类别)放入到一个列表中保存，根据欧式距离排序，去除多对一的情况，
	最终获得一个记录待配准图像特征和参考图像特征的对应情况的矩阵fit_pos，利用这个矩阵获得新的特征中心的坐标矩阵
	返回Dmatch集合--match1,新的特征中心的坐标矩阵--newkp_s,--newkp_t,记录特征的对应情况的矩阵--fit_pos 

##feature.py   
###opencv-contrib-python==3.4.2.16  
###Class Feature() 可以设置min,max，thre  

	__init__（img1,img2,shrink_num,min=None,max=None，thre=None）  
	：图像缩放，shrink_num缩放系数，  

	match_first（self)  
	：获取MSER特征和SURF描述向量并进行匹配，返回新的特征坐标--nkp_s,--nkp_t,匹配上的特征点配对的图像--Ir,字符串--lenmatch  

	match_sec(self,nkp_s,nkp_t)  
	：经过Ransac计算内点，利用内点估计仿射变换矩阵，内点数不足，保留原先的情况  

	draw_register(self,I,ww,in_s,in_t)  
	：画内点配对的图像  
	register(self)  
	：完整的特征点配准的流程（match_first,match_sec,draw_register,获取内点数)  
	图像缩放--获取MSER特征和SURF描述向量并进行匹配--经过Ransac计算内点并估计仿射变换矩阵/内点数不足保留原先的情况--画内点配对的图像--计算内点数  

##Ransac.py    

	estimate_affine(pts_s, pts_t)  
	：估计仿射变换   

	estimate_transform(pts_s, pts_t)  
	：估计刚性变换   
###Class Ransac()   

	__init__(self, K=3, threshold=1,n=0.9999)    
	：随机选点的数量K，默认阈值threshold，置信度n  

    residual_lengths(self, A, t, pts_s, pts_t)  
	：计算变换前后待配准图像特征中心与参考图像特征中心的欧式距离--residual,residual2  

    affine_matrix(kp_s, kp_t)  
	：完整的利用Ransac计算内点估计变换矩阵的流程  
	寻找内点，计算变换矩阵--利用找到的最多的内点再次最小二乘估计变换矩阵  

	ransac_fit(self, pts_s, pts_t)  
	：寻找内点，计算变换矩阵，   
    迭代次数ITER_NUM=int(((np.log10(1-self.n))/(np.log10(1-0.05*0.05*0.05))))   
    随机取三个特征对，利用这三个的位置估计仿射变换，计算经过这个估计的仿射变换后有多少个内点，随机至迭代次数，取找到的内点数最多的三个特征，当找到的内点数一致时，取内点变换前特征对的欧式距离分布更均匀（np.std更小）的结果保留  

##register.py    
###class Pre():     

	__init__(self,img1,img2,size)  
	：图像缩放，最大分辨率size，  

    crop_cross(self)   
	：去除图像中黑色的小十字，返回灰度图像  

    get_center(self,img)  
	：利用李氏交叉熵阈值图像，获取图像的质心和阈值后图像  

    gradient(self,image)  
	：计算图像梯度，prewitt算子，x,y方向的梯度竖直拼接，行是输入图像的两倍，  

    NGF(self,I1,I2)  
	：计算两张图像的NGF矩阵和值  

    pre_register(self):  
	：完整的预配准流程，  
	去除黑色小十字--取质心--图像梯度计算--图像根据质心偏差平移后旋转360度（32次）--计算NGF值并记录--取最小的NGF值对应的旋转角度--
	当归一化后的NGF值在0.01到0.02范围内，按照SSD最小的结果取旋转角度--根据旋转角度和质心偏差构建变换矩阵    