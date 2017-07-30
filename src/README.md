# Training examples on TID2013 and LIVE datasets
```
git clone https://github.com/xialeiliu/RankIQA.git
cd RankIQA
```
## Preparing Ranking and IQA datasets

The details can be found in [data](../data)

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
## Evaluation for RankIQA

```
python src/eval/Rank_eval_each_tid2013.py  # evaluation for each distortions
python src/eval/Rank_eval_all_tid2013.py   # evaluation for all distortions in tid2013
```

## Evaluation for RankIQA+FT

```
python src/eval/FT_eval_each_tid2013.py  # evaluation for each distortions
python src/eval/FT_eval_all_tid2013.py   # evaluation for all distortions in tid2013
```


## Folder introdcutions

1. [data_layer](./data_layer) includes the Python functions to read the input data for different datasets. 
2. [MyLossLayer](./MyLossLayer) contains the Python functions of our efficient back-propagation method for different datasets. 
3. [tools](./tools) provides the Python functions to calculate the evaluation matrix during training. 
4. [eval](./eval) provides the Python functions to evaluate the trained models. 

