import os
import torch
import scipy.io
import numpy as np
from multiprocess.dummy import Pool as ThreadPool


def map(func, input, thread_num=5):
    with ThreadPool(thread_num) as P:
        res = P.map(func, input)
    return res

def mat_write(total_data, mode):
    # 用于将数据写入到.mat格式，便于matlab中的数据读取
    dic_data = {}
    dic_label = {}
    for index, data in enumerate(total_data):
        data, label = data
        data = data.tolist()
        dic_data[str(index)] = data
        dic_label[str(index)] = label
    scipy.io.savemat('data_split_{}.mat'.format(mode), dic_data)
    scipy.io.savemat('label_split_{}.mat'.format(mode), dic_label) 

def mat_write_new(write_path, total_data):
    # 用于将数据写入到.mat格式，便于matlab中的数据读取
    # datas = np.zeros((total_data[0][0].shape[0],total_data[0][0].shape[1],len(total_data)))
    # labels = np.zeros()
    for index, data in enumerate(total_data):
        data, label = data
        if index == 0:
            datas = data[:, :, np.newaxis]
            labels = label[np.newaxis]
        else:
            datas = np.concatenate((datas,data[:, :, np.newaxis]),axis=-1)
            labels = np.concatenate((labels,label[np.newaxis]),axis=-1)
    scipy.io.savemat(write_path, {'datas':datas,'labels':labels})

def mat_load(load_path):
    data = scipy.io.loadmat(load_path)
    return data['datas'], data['labels']

def torch_save_model(
    model,
    save_root,
    **kwargs
    ):
    state = {'model_state_dict':model.state_dict()}
    state = dict(state, **kwargs)
    torch.save(state, save_root)
