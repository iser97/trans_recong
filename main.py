import os
import sys
import logging
import matplotlib as mpl
if os.environ.get('DISPLAY','')=='':
    print('no display found. Using non-interactive Agg backend')
    mpl.use("agg")

import numpy as np
import sklearn.metrics as skm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import HfArgumentParser

from scripts.model.transformer_single_layer import my_transformer
from scripts.data.dataset_mnist_8_8 import DatasetMnist
from scripts.config.arguments import Arguments

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
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
    confusion_matrix = skm.confusion_matrix(labels, preds)
    acc = skm.accuracy_score(labels, preds)
    recall = skm.recall_score(labels, preds, average='macro')
    f1 = skm.f1_score(np.array(labels), np.array(preds), average='macro')

    logger.info(f"confusion_matrix = {confusion_matrix}")
    logger.info(f"ACC = {acc}")
    logger.info(f"recall = {recall}")
    logger.info(f"f1 = {f1}")

def train_step(model, optimizer, loss_fn, train_loader, test_loader):
    step = make_train_step(model, loss_fn, optimizer)
    losses = []
    for epoch in range(args.n_epochs):
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
    data_dim = args.data_split_dim*args.data_split_dim
    seq_length = int(args.data_dimension**2 / data_dim)   # through the data_split_dim can split the mnist picture to sub blocks, the number of sub blocks stands for the transformers' sequence length

    tModel = my_transformer(data_dim, data_dim, seq_length, args.n_heads, data_dim, args.num_classes).to(device)
    # optimizer = optim.SGD(tModel.parameters(),lr=lr,momentum=mom)
    optimizer = optim.Adam(tModel.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    train_dataset = DatasetMnist(mode='test', split_dim=args.data_split_dim, data_dimension=args.data_dimension)
    test_dataset = DatasetMnist(mode='test', split_dim=args.data_split_dim, data_dimension=args.data_dimension)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0, 
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    train_step(tModel, optimizer, loss_fn, train_loader, test_loader)


if __name__ == '__main__':
    parser = HfArgumentParser((Arguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args, = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args, = parser.parse_args_into_dataclasses()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main()


