#!/usr/bin/env sh

TOOLS=/home/xialei/caffe/distribute/bin  # TODO  Changing to your Caffe bin

$TOOLS/caffe.bin -gpu 2 train -solver src/RankIQA/live/solver_vgg.prototxt --weights ./models/VGG_ILSVRC_16_layers.caffemodel 2>&1 | tee  models/rank_live/log_live
