import os
import torch
import scipy.io
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

def torch_save_model(
    model,
    save_root,
    **kwargs
    ):
    state = {'model_state_dict':model.state_dict()}
    state = dict(state, **kwargs)
    torch.save(state, os.path.join(save_root, "checkpoint.pth.tar"))
