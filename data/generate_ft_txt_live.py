import numpy as np
import scipy.io as sio
import os
import os.path as osp
import cv2

   
label = 'LIVE2_labels.mat'
shape = ['gblur','wn','jpeg','jp2k','fastfading']
data_path = 'data/tid2013/' 
list_file = 'ref' + '.txt'


filename = [line.rstrip('\n') for line in open(
            osp.join(data_path, list_file))]
            
ref = filename
            
gt_label = sio.loadmat(label)

shuff = np.random.permutation(range(29))
train_file = open('ft_live_train.txt', "w")
test_file = open('ft_live_test.txt', "w")

Num_tr = 23
Num_te = 29
test = False
flag = 1

for tp in shape:
    
    file_root = data_path + tp + '/'
    list_file = 'info' + '.txt'
    filename = [line.rstrip('\n') for line in open(
            osp.join(file_root, list_file))]
    S_name = [] 
    N_name = []
    scores = []
    for i in filename:
        S_name.append(i.split()[0])
        N_name.append(i.split()[1])
        scores.append(float(i.split()[2]))
        temp_label = gt_label[tp]
        temp_label = temp_label.swapaxes(1,0)
    
    a = os.listdir(tp)
    a.sort()
    
    TotalNum = temp_label.shape[0]

    for i in range(Num_tr):
        for j in range(TotalNum):
            if S_name[j] == ref[shuff[i]]:
               folder = data_path + tp + '/' + a[int(N_name[j][3:-4])-1] 
               labels = temp_label[int(N_name[j][3:-4])-1] 
               train_file.write('%s %6.2f\n' % (folder,labels))   
    
    
    for i in range(Num_tr,Num_te):
        for j in range(TotalNum):
            if S_name[j] == ref[shuff[i]]:
               folder = data_path+ tp + '/' + a[int(N_name[j][3:-4])-1] 
               if test  ==True:
                   tmp  = cv2.imread(folder)
                   cv2.imwrite('./test/' + str(flag)+'.bmp',tmp)
                   flag += 1

               labels = temp_label[int(N_name[j][3:-4])-1] 
               test_file.write('%s %6.2f\n' % (folder,labels))


train_file.close()
test_file.close()







