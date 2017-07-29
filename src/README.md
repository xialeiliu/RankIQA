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

## Folder introdcutions

1. [data_layer](./data_layer) includes the Python function to read the input data for different datasets. 
2. [MyLossLayer](./MyLossLayer) contains the Python function of our efficient back-propagation method for different datasets. 
3. [tools](./tools) provides the Python function to calculate the evaluation matrix during training. 

