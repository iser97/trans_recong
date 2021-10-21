import os
import matplotlib as mpl
if os.environ.get('DISPLAY','')=='':
    print('no display found. Using non-interactive Agg backend')
    mpl.use("agg")

import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers.utils import logging

from scripts.model.transformer_single_layer import my_transformer
from scripts.data.dataset_mnist_8_8 import DatasetMnist
logger = logging.get_logger(__name__)
torch.manual_seed(71)

def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x,y):
        # Sets model to TRAIN mode
        model.train()
        # Makes preds
        yhat = model(x)
        # Computes loss
        loss = loss_fn(yhat, y)
        # Computes gradients
        optimizer.zero_grad()
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        # Returns the loss
        return loss.item(), yhat

    # Returns the function that will be called inside the train loop
    return train_step

def test_step(model, data_loader):
    preds = []
    labels = []
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.type(torch.float32)
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(x_batch)
        pred = pred.argmax(dim=1)
        pred = pred.cpu().tolist()
        preds = preds + pred
        labels = labels + y_batch.cpu().tolist()
    confusion_matrix = sk.metrics.confusion_matrix(labels, preds)
    acc = sk.metrics.accuracy_score(labels, preds)
    recall = sk.metrics.recall_score(labels, preds, average='macro')
    f1 = sk.metrics.f1_score(np.array(labels), np.array(preds), average='macro')

    logger.info(f"confusion_matrix = {confusion_matrix}")
    logger.info(f"ACC = {acc}")
    logger.info(f"recall = {recall}")
    logger.info(f"f1 = {f1}")

def train_step(model, optimizer, loss_fn, train_loader, test_loader):
    step = make_train_step(model, loss_fn, optimizer)
    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0
        test_step(model, test_loader)
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.type(torch.float32)
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            loss, _ = step(x_batch, y_batch)
            epoch_loss += loss
        losses.append(epoch_loss)
        test_step(model, test_loader)

def main():
    tModel = my_transformer(data_dim, data_dim, seq_length, n_heads, data_dim, num_classes).to(device)
    optimizer = optim.SGD(tModel.parameters(),lr=lr,momentum=mom)
    loss_fn = nn.CrossEntropyLoss()

    train_dataset = DatasetMnist(mode='test', split_dim=data_split_dim, data_dimension=data_dimension)
    test_dataset = DatasetMnist(mode='test', split_dim=data_split_dim, data_dimension=data_dimension)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, 
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    train_step(tModel, optimizer, loss_fn, train_loader, test_loader)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ### training parameters
    batch_size = 10
    lr = 1e-4
    mom = 0.91
    n_epochs = 2000
    ### data and model parameters
    data_split_dim = 2
    data_dimension = 8 # mnist data is reshaped as 8*8
    data_dim = data_split_dim*data_split_dim
    seq_length = int(8*8 / data_dim)
    n_heads = 4
    num_classes = 10


    main()


