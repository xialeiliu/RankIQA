import numpy as np
import sys
import caffe
from scipy import stats
import cv2


caffe_root = '/home/xialei/caffe/distribute/'
sys.path.insert(0, caffe_root + 'python')

caffe.set_device(0)
caffe.set_mode_gpu()


ft = 'fc8'  # The output of network
MODEL_FILE = './src/FT/live/deploy_vgg.prototxt'
PRETRAINED_FILE = './models/ft_live/' +'my_siamese_iter_50000.caffemodel'
tp = 'FT_all' 

net = caffe.Net(MODEL_FILE, PRETRAINED_FILE, caffe.TEST)

res_dir = './results/live/'

srocc_file = open(  res_dir+ tp + '_srocc' + '.txt', "w")
lcc_file = open( res_dir + tp + '_lcc' + '.txt', "w")


test_file = './data/' + 'ft_live_test.txt'

filename = [line.rstrip('\n') for line in open(test_file)]

roidb = []
scores =[]
for i in filename:
    roidb.append(i.split()[0])
    scores.append(float(i.split()[1]))
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
print '%   LCC of mean : {}'.format(lcc)
print '% SROCC of mean: {}'.format(srocc)

srocc_file.write('%6.3f\n' % ( srocc ))
lcc_file.write('%6.3f\n' % ( lcc ))
srocc_file.close()
lcc_file.close()

srocc = stats.spearmanr(med,scores)[0] 
lcc = stats.pearsonr(med,scores)[0] 
print '%   LCC of median: {}'.format(lcc)
print '% SROCC of median: {}'.format(srocc)


