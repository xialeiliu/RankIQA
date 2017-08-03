
import numpy as np
import os
import os.path as osp
import cv2
import pdb

shape = ['gblur','wn','jpeg','jp2k','fastfading']
data = './live/' 
            
for tp in shape:
    
    file_root = data + tp + '/'
    list_file = 'info' + '.txt'
    filename = [line.rstrip('\n') for line in open(
            osp.join(file_root, list_file))]
    N_name = []
    for i in filename:
        N_name.append(i.split()[1])
    
    pdb.set_trace()
    
    for j in range(len(N_name)):
        folder = data +tp +   '/' + N_name[j]

        tmp  = cv2.imread(folder)
        cv2.imwrite(data+ tp + '/' + 'img' + str(int(N_name[j][3:-4])).zfill(3)+'.bmp',tmp)


        if int(N_name[j][3:-4])<100:            
            os.remove(folder)
    os.remove(data+tp+'/'+'Thumbs.db')








