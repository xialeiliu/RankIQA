# Training examples on TID2013 and LIVE datasets
```
git clone https://github.com/xialeiliu/RankIQA.git
cd RankIQA
```
## Requirements
1. Requirements for ```caffe ``` and  ```pycaffe ``` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html)). 
Caffe must be built with support for Python layers!

```
# In your Makefile.config, make sure to have this line uncommented
WITH_PYTHON_LAYER := 1
```
2. Requirements for GPU (Titan X (~11G of memory) is needed to train VGG).

## Preparing Ranking and IQA datasets

The details can be found in [data](../data)

## Pre-trained ImageNet VGG-16 model
Download the pre-trained [VGG-16](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md) ImageNet model and put it in the folder [models](../models).

## RankIQA

To train the RankIQA models on tid2013 dataset:

```
./src/RankIQA/tid2013/train_vgg.sh
```

To train the RankIQA models on LIVE dataset:

```
./src/RankIQA/live/train_vgg.sh
```

## FT

To train the RankIQA+FT models on tid2013 dataset:

```
./src/FT/tid2013/train_vgg.sh
```
To train the RankIQA+FT models on LIVE dataset:

```
./src/FT/live/train_live.sh
```
## Evaluation for RankIQA on tid2013:

```
python src/eval/Rank_eval_each_tid2013.py  # evaluation for each distortions in tid2013
python src/eval/Rank_eval_all_tid2013.py   # evaluation for all distortions in tid2013
```

## Evaluation for RankIQA+FT on tid2013:

```
python src/eval/FT_eval_each_tid2013.py  # evaluation for each distortions in tid2013
python src/eval/FT_eval_all_tid2013.py   # evaluation for all distortions in tid2013
```

## Evaluation for RankIQA on LIVE:

```
python src/eval/Rank_eval_all_live.py   # evaluation for all distortions in LIVE
```

## Evaluation for RankIQA+FT on LIVE:

```
python src/eval/FT_eval_all_live.py   # evaluation for all distortions in LIVE
```

## Folder introdcutions

1. [data_layer](./data_layer) includes the Python functions to read the input data for different datasets. 
2. [MyLossLayer](./MyLossLayer) contains the Python functions of our efficient back-propagation method for different datasets. 
3. [tools](./tools) provides the Python functions to calculate the evaluation matrix during training. 
4. [eval](./eval) provides the Python functions to evaluate the trained models. 

