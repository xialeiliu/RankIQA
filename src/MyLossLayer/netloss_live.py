import caffe
import numpy as np
import pdb

class MyLossLayer(caffe.Layer):
    """Layer of Efficient Siamese loss function."""

    def setup(self, bottom, top):
        self.margin = 1
        print '*********************** SETTING UP'
        pass

    def forward(self, bottom, top):
        """The forward """
        self.Num = 0
        batch = 2
        level = 6
        dis = 4
        SepSize = batch*level
        self.dis = []
        # for the first
        for k in range(dis):
            for i in range(SepSize*k,SepSize*(k+1)-batch):
                for j in range(SepSize*k + int((i-SepSize*k)/batch+1)*batch,SepSize*(k+1)):
                    self.dis.append(bottom[0].data[i]-bottom[0].data[j])
                    self.Num +=1
          
        self.dis = np.asarray(self.dis)        
        self.loss = np.maximum(0,self.margin-self.dis)    # Efficient Siamese forward pass of hinge loss

        top[0].data[...] = np.sum(self.loss)/bottom[0].num


    def backward(self, top, propagate_down, bottom):
        """The parameters here have the same meaning as data_layer"""
        batch=2
        index = 0
        level = 6
        dis = 4
        SepSize = batch*level
        self.ref= np.zeros(bottom[0].num,dtype=np.float32)
        for k in range(dis):
            for i in range(SepSize*k,SepSize*(k+1)-batch):
                for j in range(SepSize*k + int((i-SepSize*k)/batch+1)*batch,SepSize*(k+1)):
                    if self.loss[index]>0:
                        self.ref[i] += -1
                        self.ref[j] += +1
                    index +=1

        # Efficient Siamese backward pass        
        bottom[0].diff[...]= np.reshape(self.ref,(bottom[0].num,1))/bottom[0].num
        
                
    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""

        top[0].reshape(1)
 
