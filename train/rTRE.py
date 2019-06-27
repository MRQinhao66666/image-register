import numpy as np
import cv2
import os
current_path = os.getcwd()
image_path = os.path.join(current_path, 'training','image')
label_path = os.path.join(current_path, 'training','label')
result_path =os.path.join(current_path)
image_dir1 = sorted(os.listdir(image_path),key=str.lower,reverse=False)[:]
result_filename = os.path.join(result_path, 'labelrTRE.txt')

def main():
    with open(result_filename, 'a', encoding='utf-8') as f1:
        f1.writelines('{0:<80}{1:<20}{2:<15}{3:<15}\n'.format('name', 'rTRE(median_rTRE)', 'max_rTRE', 'mean_rTRE'))
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
            file_target_path = os.path.join(image_path2,str(dir2)+'.jpg')
            file_target=os.path.basename(file_target_path)
            file = [x for x in file_image if x !=file_target]
            label_target_path=os.path.join(readlabel,m,str(file_target)[:-4]+'.csv')
            for idx in file:
                file_source_path=os.path.join(image_path2,str(idx))
                label_source_path=os.path.join(readlabel,m,str(os.path.basename(idx)[:-4])+'.csv')
                data_source = (np.loadtxt(label_source_path, dtype=np.str, delimiter=","))[1:, 1:].astype(np.float)
                data_target = (np.loadtxt(label_target_path, dtype=np.str, delimiter=","))[1:, 1:].astype(np.float)
                save_name = os.path.join(os.path.basename(file_source_path)[:-4] + 'to' + str(file_target)[:-4])
                # img_source = cv2.imread(source_path, 0)
                img_target = cv2.imread(file_target_path, 0)
                rows, cols = img_target.shape
                ###   compute the TRE of all points
                TRE = np.linalg.norm((data_target-data_source),axis=1, ord=2)
                ### normalized TRE ,get rTRE
                r = (np.sqrt(np.power(rows, 2) + np.power(cols, 2)))
                rTRE = TRE / r
                rTRE = np.sort(rTRE)
                k = int((rTRE.size) / 2)
                median_rTRE = rTRE[k]
                max_rTRE = max(rTRE)
                mean_rTRE = np.mean(rTRE)
                with open(result_filename, 'a', encoding='utf-8') as f1:
                    f1.writelines('{0:<80s}{1:<20n}{2:<15n}{3:<15.4f}\n'.format(save_name,median_rTRE,max_rTRE,mean_rTRE))

if __name__ == '__main__':
    main()

#########fileset rename
# import sys
# import os
# import shutil
# import random
#
# pwd = os.getcwd()
# A ="./test"
# L = os.listdir(A)
# # f = open("rename.txt", "w")
# for dirname in L:
#         # filename = files[0]
#         a=os.path.join(A,dirname)
#         sfile = os.listdir(a)[0]
#         sdir = os.path.join(a,sfile)
#         # mfile =  os.listdir(a)[-1]
#         # mdir = os.path.join(a, mfile)
#         # smallfile =os.path.join(a,"small")
#         mediumfile =os.path.join(a,"medium")
#         # os.rename(sdir, smallfile)
#         os.rename(sdir, mediumfile)
