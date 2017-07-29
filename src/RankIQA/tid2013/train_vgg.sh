#!/usr/bin/env sh

TOOLS=/home/xialei/caffe/distribute/bin  #TODO Chaning to your Caffe bin

$TOOLS/caffe.bin -gpu 0 train -solver src/RankIQA/tid2013/solver_vgg.prototxt --weights ./models/VGG_ILSVRC_16_layers.caffemodel 2>&1 | tee  models/rank_tid2013/log_tid2013
