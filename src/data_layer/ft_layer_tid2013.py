import cv2
import caffe
import numpy as np
import multiprocessing as mtp
import pdb
import os.path as osp

class DataLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""
    def setup(self, bottom, top):

        self._name_to_top_map = {}
        self._name_to_top_map['data'] = 0
        self._name_to_top_map['label'] = 1
        # === Read input parameters ===
        self.workers= mtp.Pool(10)
        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)

        # Check the paramameters for validity.
        check_params(params)

        # store input as class variables
        self.batch_size = params['batch_size']
        self.pascal_root = params['pascal_root']
        self.im_shape = params['im_shape']
        # get list of image indexes.
        list_file = params['split'] + '.txt'
        filename = [line.rstrip('\n') for line in open(
            osp.join(self.pascal_root, list_file))]
        self._roidb = []
        self.scores =[]
        for i in filename:
            self._roidb.append(i.split()[0])
            self.scores.append(float(i.split()[1]))
        self._perm = None
        self._cur = 0
        self._shuffle_roidb_inds()
        
        top[0].reshape(
            self.batch_size, 3, params['im_shape'][0], params['im_shape'][1])
     
        top[1].reshape(self.batch_size, 1)    
    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        # TODO(rbg): remove duplicated code
        if self._cur + self.batch_size  >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + self.batch_size ]
        self._cur += self.batch_size 
        return db_inds
    def get_minibatch(self,minibatch_db):
        """Given a roidb, construct a minibatch sampled from it."""
       
        jobs =self.workers.map(preprocess,minibatch_db)
        #print len(jobs)
        index = 0
        images_train = np.zeros([self.batch_size,3,224,224],np.float32)
        #pdb.set_trace()
        for index_job in range(len(jobs)):
            images_train[index,:,:,:] = jobs[index_job]
            index += 1 
                       
       
        blobs = {'data': images_train}
        return blobs

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        

        db_inds = self._get_next_minibatch_inds()
        minibatch_db = []
        for i in range(len(db_inds)):
            minibatch_db.append(self._roidb[int(db_inds[i])])
        #minibatch_db = [self._roidb[i] for i in db_inds]
        #print minibatch_db
        scores = []
        for i in range(len(db_inds)):
            scores.append(self.scores[int(db_inds[i])])
        ind = np.argsort(scores)
        ind_minibatch_db = []
        for i in range(len(ind)-1,-1,-1):
            ind_minibatch_db.append(minibatch_db[ind[i]])
        blobs = self.get_minibatch(ind_minibatch_db)
        blobs ['label'] =np.asarray(sorted(scores,reverse=True))
        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def preprocess(data):

    im = np.asarray(cv2.imread('tid2013/'+ data))
    x =  im.shape[0]
    y = im.shape[1]
    x_p = np.random.randint(x-224,size=1)[0]
    y_p = np.random.randint(y-224,size=1)[0] 
    #print x_p,y_p   
    images = im[x_p:x_p+224,y_p:y_p+224,:].transpose([2,0,1])
    #print images.shape
    return images

def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    assert 'split' in params.keys(
    ), 'Params must include split (train, val, or test).'

    required = ['batch_size', 'pascal_root', 'im_shape']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)
