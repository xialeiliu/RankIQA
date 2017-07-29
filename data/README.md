
## Datasets preparation

We provide some examples about how to prepare the structure of data in this folder.

This is essential because the data structure is related to the [data reader function](../src/data_layer) and the [loss function](../src/MyLossLayer).

### Ranking datasets preparation

To train the RankIQA model on generated rank dataset, the data structure should be like [tid2013_test.txt](./tid2013_test.txt).

### IQA datasets preparation

To train the RankIQA+FT model finetuned on tid2013 dataset, the data structure should be like [ft_tid2013_test.txt](./ft_tid2013_test.txt) 

