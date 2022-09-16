import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from saif.ml_utils.activations import MONOTONIC_FUNCS as mfuncs


RNN_MODULES = {
    'lstm' : nn.LSTM,
    'rnn' : nn.RNN,
    'gru' : nn.GRU
}

class SeismicDataModule(pl.LightningDataModule):
    def __init__(self, Xt, Yt, Xv, Yv, Xtest, Ytest):
        super().__init__()
        
        self.Xt = Xt
        self.Yt = Yt
        self.Xv = Xv
        self.Yv = Yv
        self.Xtest = Xtest
        self.Ytest = Ytest

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass
        
    def train_dataloader(self):
        train_dataset = torch.utils.data.TensorDataset(self.Xt, self.Yt)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=1, 
            shuffle=False
        )
        
        return train_loader
    
    def val_dataloader(self):
        val_dataset = torch.utils.data.TensorDataset(self.Xv, self.Yv)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=1, 
            shuffle=False
        )
        
        return val_loader

    def test_dataloader(self):
        test_dataset = torch.utils.data.TensorDataset(self.Xtest, self.Ytest)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=1, 
            shuffle=False
        )
        
        return test_loader

class RNNForecaster(pl.LightningModule):
    def __init__(
            self, 
            input_size, 
            hidden_size, 
            num_layers, 
            dropout, 
            learning_rate,
            criterion,
            monotonic_fn='sqrt',
            use_bias=False,
            rnn_type='rnn'
        ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.monotonic_fn = monotonic_fn
        self.use_bias = use_bias
        self.activation = mfuncs.get(monotonic_fn, lambda x : x)

        self.rnn = RNN_MODULES.get(rnn_type)(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers, 
            dropout=dropout, 
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, 1, bias=self.use_bias)
        if self.use_bias:
            self.linear.bias.data = torch.zeros_like(self.linear.bias.data)
        self.linear.weight.data = self.linear.weight.data / 1000
    
    def forward(self, x, h=None):
        # lstm_out = (batch_size, seq_len, hidden_size)
        out, h = self.rnn(x)
        out = self.linear(out)

        out = self.activation(out)

        if self.monotonic_fn != 'no_func':
            out = out.cumsum(1)
        
        return out, h
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        return loss