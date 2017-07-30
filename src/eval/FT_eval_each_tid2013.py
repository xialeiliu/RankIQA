import numpy as np
import sys
import caffe
from scipy import stats
import cv2


caffe_root = '/home/xialei/caffe/distribute/'
sys.path.insert(0, caffe_root + 'python')
caffe.set_device(4)
caffe.set_mode_gpu()

ft = 'fc8'  # The output of network
MODEL_FILE = './src/FT/tid2013/deploy_vgg.prototxt'
PRETRAINED_FILE = './models/ft_rank_tid2013/' +'my_siamese_iter_10000.caffemodel'
tp = 'FT_each' 
     
net = caffe.Net(MODEL_FILE, PRETRAINED_FILE, caffe.TEST)

res_dir = './results/tid2013/'

srocc_file = open(  res_dir+ tp + '_srocc' + '.txt', "w")
lcc_file = open( res_dir + tp + '_lcc' + '.txt', "w")

test_file = './data/' + 'ft_tid2013_test.txt'
filename = [line.rstrip('\n') for line in open(test_file)]


for tp in range(1,25):
    
    roidb = []
    scores =[]
    tmp_labels=[]
    tmp_dir=[]
    for i in filename:
        tmp_dir.append(i.split()[0])
        tmp_labels.append(float(i.split()[1]))
    tmp_labels = np.asarray(tmp_labels)
    #print scores
    
    for i in range(len(tmp_labels)):
        #print tmp_dir[i][-8:-6]
        if int(tmp_dir[i][-8:-6]) == int(tp):
            
            roidb.append(tmp_dir[i])
            scores.append(tmp_labels[i]) 
            
    scores = np.asarray(scores)
    
    
    Num_Patch = 30
    Num_Image = len(scores)
    feat = np.zeros([Num_Image,Num_Patch])
    pre = np.zeros(Num_Image)
    med = np.zeros(Num_Image)

    for i in range(Num_Image):
        directory = roidb[i]
        im = np.asarray(cv2.imread(directory))
        for j in range(Num_Patch):   
            x =  im.shape[0]
            y = im.shape[1]
            x_p = np.random.randint(x-224,size=1)[0]
            y_p = np.random.randint(y-224,size=1)[0]  
            temp = im[x_p:x_p+224,y_p:y_p+224,:].transpose([2,0,1])

            out = net.forward_all(data=np.asarray([temp]))
            feat[i,j] = out[ft][0]
            pre[i] += out[ft][0]
        pre[i] /= Num_Patch
        med [i] = np.median(feat[i,:])
    srocc = stats.spearmanr(pre,scores)[0] 
    lcc = stats.pearsonr(pre,scores)[0] 
    print str(tp) + '%   LCC of mean: {}'.format(lcc)
    print str(tp) + '% SROCC of mean: {}'.format(srocc)
    
    srocc_file.write('%6.3f\n' % ( srocc ))
    lcc_file.write('%6.3f\n' % ( lcc ))
    
    srocc = stats.spearmanr(med,scores)[0] 
    lcc = stats.pearsonr(med,scores)[0] 
    print str(tp) +'%   LCC of median: {}'.format(lcc)
    print str(tp) + '% SROCC of median: {}'.format(srocc)


srocc_file.close()
lcc_file.close()

