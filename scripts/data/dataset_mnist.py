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

from scripts.utils.utils import map, mat_write, mat_write_new, mat_load

class DatasetMnist(Dataset):
    '''
    Parameters:
        mode: train or test
    '''
    def __init__(self, mode='train', split_dim=4, data_dimension=8, doSplit = True) -> None:
        super(DatasetMnist).__init__()
        self.split_dim = split_dim
        self.data_dimension = data_dimension
        if self.data_dimension % self.split_dim != 0:  
            raise ValueError('split dimension not match the data dimension')
        self.split_iter = int(self.data_dimension // self.split_dim)
        if doSplit:
            if mode == 'train':
                self.data = open('./cache/train_data_{}.txt'.format(data_dimension), mode='r', encoding='utf-8')
                self.label = open('./cache/train_label_{}.txt'.format(data_dimension), mode='r', encoding='utf-8')
            elif mode == 'test':
                self.data = open('./cache/test_data_{}.txt'.format(data_dimension), mode='r', encoding='utf-8')
                self.label = open('./cache/test_label_{}.txt'.format(data_dimension), mode='r', encoding='utf-8')
            else:
                raise ValueError('{} mode not implemented'.format(mode))
            self.data = np.array(json.loads(self.data.readlines()[0]))
            self.label = np.array(json.loads(self.label.readlines()[0]))
        else:
            if mode == 'train':
                self.data, self.label = mat_load('./cache/train_data_{}.mat'.format(data_dimension))
            elif mode == 'test':
                self.data, self.label = mat_load('./cache/test_data_{}.mat'.format(data_dimension))
            else:
                raise ValueError('{} mode not implemented'.format(mode))
            
        assert len(self.data) == len(self.label)
        self.sample_size = len(self.label) 
        self.data_split()
        # 如果需要将数据存储为.mat格式，调用该函数
        # write_path = '{}_data_split_{}.mat'.format(mode,data_dimension)
        # mat_write_new(write_path, self.data)
        # mat_write(self.data, mode)

        # w_data = open('{}_data.txt'.format(mode), mode='w', encoding='utf-8')
        # w_label = open('{}_label.txt'.format(mode), mode='w', encoding='utf-8')
        # self.data_w = [item[0].tolist() for item in self.data]
        # self.label_w = [item[1] for item in self.data]
        # w_data.writelines(json.dumps(self.data_w))
        # w_label.writelines(json.dumps(self.label_w))

    
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

    def __len__(self) -> int:
        return self.sample_size
    
    def __getitem__(self, index: int):
        if index >= self.sample_size:
            raise IndexError
        data = self.data[index][0]
        label = self.data[index][1]
        return data, label

class DatasetMnist_noSplit(Dataset):
    '''
    Parameters:
        mode: train or test
    '''
    def __init__(self, mode='train',data_dimension=8) -> None:
        super(DatasetMnist).__init__()
        if mode == 'train':
            self.data = open('./cache/train_data_{}.txt'.format(data_dimension), mode='r', encoding='utf-8')
            self.label = open('./cache/train_label_{}.txt'.format(data_dimension), mode='r', encoding='utf-8')
        elif mode == 'test':
            self.data = open('./cache/test_data_{}.txt'.format(data_dimension), mode='r', encoding='utf-8')
            self.label = open('./cache/test_label_{}.txt'.format(data_dimension), mode='r', encoding='utf-8')
        else:
            raise ValueError('{} mode not implemented'.format(mode))
        self.data = np.array(json.loads(self.data.readlines()[0]))
        self.label = np.array(json.loads(self.label.readlines()[0]))
        assert len(self.data) == len(self.label)
        self.sample_size = len(self.label) 

        ### 如果需要将数据存储为.mat格式，调用该函数
        # mat_write(self.data, mode)
        

    def __len__(self) -> int:
        return self.sample_size
    
    def __getitem__(self, index: int):
        if index >= self.sample_size:
            raise IndexError
        data = self.data[index]
        label = self.label[index]
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


