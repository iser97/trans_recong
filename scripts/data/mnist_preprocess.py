import os
import sys
CUR_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append('{}/../../'.format(CUR_ROOT))
import torch
import json
from typing import Optional
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.transforms.transforms import Resize
from transformers import HfArgumentParser
from scripts.config.arguments import Arguments

def data_preprocess(args):
    mnist_data_root = '/home/zjh/python_program/data/' # change by yourself situation
    batch_size = 1000
    data_dimension = args.data_dimension
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(mnist_data_root, train=True, download=True,
                    transform=transforms.Compose([
                        transforms.Resize([data_dimension, data_dimension]),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=batch_size, shuffle=False, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(mnist_data_root, train=False, transform=transforms.Compose([
                        transforms.Resize([data_dimension, data_dimension]),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=batch_size, shuffle=False, **kwargs)

    train_data = []
    test_data = []
    train_label = []
    test_label = []
    for batch_idx, (data, target) in enumerate(train_loader):
        target_lst = target.tolist()
        train_label += target_lst
        data = torch.squeeze(data)
        data = data.reshape(batch_size, data_dimension*data_dimension)
        train_data.append(data)
    train_data = torch.stack(train_data)
    train_data = train_data.reshape(-1, data_dimension*data_dimension)
    train_data = train_data.tolist()
    with open(os.path.join(args.data_root, 'train_data_{}.txt'.format(str(data_dimension))), mode='w', encoding='utf-8') as w:
        train_data = json.dumps(train_data)
        w.writelines(train_data)
    with open(os.path.join(args.data_root, 'train_label_{}.txt'.format(str(data_dimension))), mode='w', encoding='utf-8') as w:
        train_label = json.dumps(train_label)
        w.writelines(train_label)

    for batch_idx, (data, target) in enumerate(test_loader):
        target_lst = target.tolist()
        test_label += target_lst
        data = torch.squeeze(data)
        data = data.reshape(batch_size, data_dimension*data_dimension)
        test_data.append(data)
    test_data = torch.stack(test_data)
    test_data = test_data.reshape(-1, data_dimension*data_dimension)
    test_data = test_data.tolist()
    with open(os.path.join(args.data_root, 'test_data_{}.txt'.format(str(data_dimension))), mode='w', encoding='utf-8') as w:
        test_data = json.dumps(test_data)
        w.writelines(test_data)
    with open(os.path.join(args.data_root, 'test_label_{}.txt'.format(str(data_dimension))), mode='w', encoding='utf-8') as w:
        test_label = json.dumps(test_label)
        w.writelines(test_label)

if __name__ == '__main__':
    parser = HfArgumentParser((Arguments))
    args, = parser.parse_args_into_dataclasses()

    ### check whether the preprocessed data is exists
    check_root = os.path.join(args.data_root, 'train_data_{}'.format(args.data_dimension))
    if os.path.exists(check_root):
        print("Preprocessed data already exists")
    else:
        data_preprocess(args)