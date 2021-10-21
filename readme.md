# Transformers single layer training on Mnist

## file dir

cache: root for saving data, the 'test_data.txt', 'train_data.txt', 'test_label.txt', 'train_label.txt' is save in this directory.

scripts: total scripts

training and test run the following command
```bash
python main.py --lr 0.0001 --batch_size 256 --n_epochs 200 --data_split_dim 4 --data_dimension 8 --n_heads 4
```