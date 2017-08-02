
## Datasets preparation

We provide some examples about how to prepare the structure of data in this folder.

This is essential because the data structure is related to the [data reader function](../src/data_layer) and the [loss function](../src/MyLossLayer).

1. Download the [LIVE2](http://live.ece.utexas.edu/research/quality/subjective.htm) dataset and rename it as [live](./live) folder. Rename the image name in each distortion folder from "imgx.bmp" and "imgxx.bmp" to "img0xx.bmp", which allows us to read the data by order.
```
python rename_image_names.py
```
2. Download the [TID2013](http://www.ponomarenko.info/tid2013.htm) dataset and rename it as [tid2013](./tid2013) folder.
3. Download the [Waterloo](https://ece.uwaterloo.ca/~zduanmu/cvpr16_gmad/) or Validation set of [Places2](http://places2.csail.mit.edu/) datasets, and put it in [rank_live](./rank_live) and [rank_tid2013](./rank_tid2013) folders.
4. Follow the instructions of [Waterloo](https://ece.uwaterloo.ca/~zduanmu/cvpr16_gmad/) to generate 4 distortions (JPEG, JP2K, Gblur and GN) at five levels in [rank_live](./rank_live). The code to generate 17 distortions for tid2013 dataset will be publicly available in the future.

### Ranking datasets preparation

To train the RankIQA model on generated rank dataset, the data structure should be like [tid2013_train.txt](./tid2013_train.txt).

Run the code to generate the live_train.txt and live_test.txt files

```
cd data
python generate_rank_txt_live.py 
```

Run the code to generate the tid2013_train.txt and tid2013_test.txt files

```
python generate_rank_txt_tid2013.py 
```

### IQA datasets preparation

To train the RankIQA+FT model finetuned on tid2013 dataset, the data structure should be like [ft_tid2013_train.txt](./ft_tid2013_train.txt).

Run the code to generate the ft_live_train.txt and ft_live_test.txt files

```
python generate_ft_txt_live.py 
```

Run the code to generate the ft_tid2013_train.txt and ft_tid2013_test.txt files

```
python generate_ft_txt_tid2013.py
cd ..
```
