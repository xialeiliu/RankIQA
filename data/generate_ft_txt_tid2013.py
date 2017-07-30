
import numpy as np
import scipy.io as sio
import os
import os.path as osp

data_dir = 'tid2013/'

file_root = data_dir + 'distorted_images' + '/'
list_file = data_dir + 'mos_with_names' + '.txt'


filename = [line.rstrip('\n') for line in open(
            osp.join(list_file))]
S_name = [] 
scores = []
for i in filename:
    S_name.append(i.split()[1])
    scores.append(float(i.split()[0]))
            
ref = S_name

TotalNum = len(ref)
shuff = np.random.permutation(range(1,25))

Num_tr = 19  # ~80% of reference images as training samples
Num_te = 5   # ~20% of reference images as testing samples

train_file = open('ft_tid2013'+'_train.txt', "w")
test_file = open('ft_tid2013'+'_test.txt', "w")

shuff_txt=np.random.permutation(range(TotalNum))    
    
for i in shuff_txt:
        #if ref[i][4:6]==tp:
        for j in range(Num_tr):
            if int(ref[i][1:3]) == shuff[j]:
               folder = data_dir + 'distorted_images/' + ref[i]
               labels = scores[i]
               train_file.write('%s %6.2f\n' % (folder,labels))


    

for i in shuff_txt:
        #if ref[i][4:6]==tp:
        for j in range(Num_tr,Num_tr+Num_te):
            if int(ref[i][1:3]) == shuff[j]:
               folder = data_dir + 'distorted_images/' + ref[i]
               labels = scores[i]
               test_file.write('%s %6.2f\n' % (folder,labels))

train_file.close()   
test_file.close()
