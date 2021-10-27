'''
构造mnist数据的dataset与dataloader，首先要基于mnist_8_8.py文件生成数据
'''
import os
import sys
CUR_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append('{}/../../'.format(CUR_ROOT))
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from scripts.utils.utils import map, mat_write

class DatasetMnist(Dataset):
    '''
    Parameters:
        mode: train or test
    '''
    def __init__(self, mode='train', split_dim=4, data_dimension=8) -> None:
        super(DatasetMnist).__init__()
        self.split_dim = split_dim
        self.data_dimension = data_dimension
        if self.data_dimension % self.split_dim != 0:  
            raise ValueError('split dimension not match the data dimension')
        self.split_iter = int(self.data_dimension // self.split_dim)

        if mode == 'train':
            self.data = open('./cache/train_data.txt', mode='r', encoding='utf-8')
            self.label = open('./cache/train_label.txt', mode='r', encoding='utf-8')
        elif mode == 'test':
            self.data = open('./cache/test_data.txt', mode='r', encoding='utf-8')
            self.label = open('./cache/test_label.txt', mode='r', encoding='utf-8')
        else:
            raise ValueError('{} mode not implemented'.format(mode))
        self.data = np.array(json.loads(self.data.readlines()[0]))
        self.label = json.loads(self.label.readlines()[0])
        assert len(self.data) != self.label
        self.sample_size = len(self.label) 
        self.data_split()
        ### 如果需要将数据存储为.mat格式，调用该函数
        mat_write(self.data, mode)

    
    def data_split(self,):
        def data_split(input):
            data, label = input[0], input[1]
            cur_res = []
            data = np.reshape(data, (self.data_dimension, self.data_dimension))
            data_row_split = np.split(data, self.split_iter, axis=0)
            for item in data_row_split:
                cur_res += np.split(item, self.split_iter, axis=1)
            cur_res = np.stack(cur_res)
            cur_res = np.reshape(cur_res, (-1, self.split_dim**2))
            return cur_res, label
        total_res = map(data_split, zip(self.data, self.label), thread_num=16)
        self.data = total_res
        # for data in total_res:
        #     data = np.reshape(data, (8, 8))
        #     plt.imsave('t.jpg', data)

        # for idx in range(self.sample_size):
        #     data = self.data[idx]
        #     cur_res = []
        #     data = np.reshape(data, (self.data_dimension, self.data_dimension))
        #     data_row_split = np.split(data, self.split_iter, axis=0)
        #     for item in data_row_split:
        #         cur_res += np.split(item, self.split_iter, axis=1)
        #     cur_res = np.stack(cur_res)
        #     cur_res = np.reshape(cur_res, (-1, self.split_dim**2))
        #     self.new_data.append(cur_res)
        # self.data = self.new_data

    def __len__(self) -> int:
        return self.sample_size
    
    def __getitem__(self, index: int):
        if index >= self.sample_size:
            raise IndexError
        data = self.data[index][0]
        label = self.data[index][1]
        return data, label

if __name__ == '__main__':
    d = DatasetMnist(mode='train')
    loader = DataLoader(d, batch_size=10, shuffle=False, num_workers=0, drop_last=False)
    for data, label in loader:
        print(label[0])
    
    ### code for testing split
    # arr = np.array(range(0, 64))
    # arr = np.reshape(arr, (8, 8))
    # arr = np.split(arr, 4, 0)
    # res = []
    # for item in arr:
    #     res += np.split(item, 4, 1)
    # print(res)


