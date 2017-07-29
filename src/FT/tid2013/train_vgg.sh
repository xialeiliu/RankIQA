#!/usr/bin/env sh

TOOLS=/home/xialei/caffe/distribute/bin #TODO Changing to your Caffe bin

$TOOLS/caffe.bin train --gpu 4 --solver=src/FT/tid2013/solver_vgg.prototxt --weights ./models/rank_tid2013/my_siamese_iter_50000.caffemodel  2>&1 | tee  models/ft_rank_tid2013/log_tid2013
