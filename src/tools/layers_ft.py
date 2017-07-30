import caffe
from scipy import stats
import numpy as np
#import ipdb
class AngularErrorLayer(caffe.Layer):
    """Layer that computes SROCC and LCC on batch."""

    def setup(self, bottom, top):
        print '*********************** SETTING UP'
        pass

    def forward(self, bottom, top):
        """Compute the SROCC and LCC and output them to top."""
        #ipdb.set_trace()
        testPreds = bottom[0].data
        testPreds = np.reshape(testPreds,testPreds.shape[0])
        testLabels = bottom[1].data
        testLabels = np.reshape(testLabels,testLabels.shape[0])
        top[0].data[...] = stats.spearmanr(testPreds, testLabels)[0]
        top[1].data[...] = stats.pearsonr(testPreds, testLabels)[0]

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        top[0].reshape(1)
        top[1].reshape(1)
 
