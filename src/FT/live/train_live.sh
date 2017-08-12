#!/usr/bin/env sh

TOOLS=/home/xialei/caffe/distribute/bin #TODO Changing to your Caffe bin

$TOOLS/caffe.bin train --gpu 0 --solver=src/FT/live/solver_live.prototxt --weights ./models/rank_live_20/my_siamese_iter_20000.caffemodel  2>&1 | tee  models/ft_live/log_live
