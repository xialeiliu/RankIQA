import numpy as np
import scipy.io as sio
import os


# folder includes all distortion types of ranking data for tid2013
folder = ['JPEG','GN','GB','JP2K']

dir_rank = './data/rank_live/'   # Change to your data folder 

train_file = open('live'+'_train.txt', "w")
test_file = open('live'+'_test.txt', "w")

pristine = 'pristine_images'  # The folder of reference images to generate different distortions

real = os.listdir(dir_rank + pristine)
real.sort()
Num = len(real)
Tr_num = int(Num*0.8)

FileT_p = []    # To save the image names
scores_p = []   # To save the distortion levels

for i in real:
    FileT_p.append(dir_rank + pristine + '/' + i )
    scores_p.append(5)

shuff_p = range(Num)
#shuff_p = np.random.permutation(range(Num))         # To decide shuffle the data or not   

for t in folder:
    

    DisType = os.listdir(dir_rank + t)
    DisType.sort(reverse  = True)

    ind =0
    dis_level = 5      # dis_level +1 = Level of distortion can be chose by changing this variable 
    shuff = range(Num)
    #shuff = np.random.permutation(range(Num))   
    for i in DisType[0:dis_level]:
        fileN = os.listdir(dir_rank + t+'/'+i)
        fileN.sort()
        
        FileT = []
        scores = []
        for j in range(len(fileN)):
            FileT.append(dir_rank + t + '/' + i +'/'+ fileN[j]) 
            scores.append(ind)
        for i in range(Tr_num):
            train_file.write('%s %6.2f\n' % ( FileT[shuff[i]],scores[shuff[i]]))
        for i in range(Tr_num,Num):
            test_file.write('%s %6.2f\n' % ( FileT[shuff[i]],scores[shuff[i]])) 
        ind += 1
    
    for i in range(Tr_num):
        train_file.write('%s %6.2f\n' % ( FileT_p[shuff_p[i]],scores_p[shuff_p[i]]))
    for i in range(Tr_num,Num):
        test_file.write('%s %6.2f\n' % ( FileT_p[shuff_p[i]],scores_p[shuff_p[i]]))  

train_file.close()
test_file.close()



