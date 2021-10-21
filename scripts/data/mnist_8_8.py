import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.transforms.transforms import Resize

kwargs = {'num_workers': 1, 'pin_memory': True}
batch_size = 1000
mnist_data_root = '/home/zjh/python_program/data/' # change by yourself situation
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(mnist_data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Resize([8, 8]),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
    batch_size=batch_size, shuffle=False, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(mnist_data_root, train=False, transform=transforms.Compose([
                    transforms.Resize([8, 8]),
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
    data = data.reshape(batch_size, 8*8)
    train_data.append(data)
train_data = torch.stack(train_data)
train_data = train_data.reshape(-1, 8*8)
train_data = train_data.tolist()
with open('../../cache/train_data.txt', mode='w', encoding='utf-8') as w:
    train_data = json.dumps(train_data)
    w.writelines(train_data)
with open('../../cache/train_label.txt', mode='w', encoding='utf-8') as w:
    train_label = json.dumps(train_label)
    w.writelines(train_label)

for batch_idx, (data, target) in enumerate(test_loader):
    target_lst = target.tolist()
    test_label += target_lst
    data = torch.squeeze(data)
    data = data.reshape(batch_size, 8*8)
    test_data.append(data)
test_data = torch.stack(test_data)
test_data = test_data.reshape(-1, 8*8)
test_data = test_data.tolist()
with open('../../cache/test_data.txt', mode='w', encoding='utf-8') as w:
    test_data = json.dumps(test_data)
    w.writelines(test_data)
with open('../../cache/test_label.txt', mode='w', encoding='utf-8') as w:
    test_label = json.dumps(test_label)
    w.writelines(test_label)