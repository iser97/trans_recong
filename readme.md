# Transformers single layer training on Mnist

## data preprocess
At first, we need run script `./scripts/data/mnist_preprocess.py` to get pre-processed data

```bash
python ./scripts/data/mnist_preprocess.py --data_size 28
```

## file dir

cache: root for saving data, the 'test_data.txt', 'train_data.txt', 'test_label.txt', 'train_label.txt' is saved in this directory.

scripts: total scripts

更改模型类别可以通过参数model_type进行更改， model_type=transformer or model_type=linear

training and test run the following command
```bash
python main.py --lr 0.001 --batch_size 256 --n_epochs 10 --data_split_dim 4 --data_dimension 8 --n_heads 1 --model_type transformer --gama_scale 0.001
```



