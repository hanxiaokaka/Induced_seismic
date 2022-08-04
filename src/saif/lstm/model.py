import numpy as np
import torch
from torch import nn
from typing import Callable

class ShallowRegLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float,
                 monotonic_fn: Callable = lambda x : x) -> None:
        '''
        Shallow LSTM model for regression

        Parameters:
        -------------
        input_size: (int) Number of expected features in input
        hidden_size: (int) Number of features in the hidden state h
        num_layers: (int) Number of recurrent layers
        dropout: (float) Dropout probability on LSTM layers excluding the final layer
        monotonic_fn: (function) Inductive bias applied on model output to enforce monotonicity
        '''
        super(ShallowRegLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.monotonic_fn = monotonic_fn

        # By default, nn.LSTM() assumes the batch number to be the second index.
        # We use batch_first=True to conform with our data set structure.
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            dropout=self.dropout, batch_first=True)
        self.linear = nn.Linear(in_features=self.hidden_size*self.num_layers, out_features=1)

    def forward(self, x) -> float:
        '''
        Forward pass of input data through the model
        '''
        # Shape of x = (batch size, input_size, number of input features)
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        # Shape of hn = (num_layers, batch size, hidden_size)
        out = self.linear(hn.permute(1,0,2).flatten(start_dim=1, end_dim=-1)).flatten()
        # Enforce monotonicity.
        out = self.monotonic_fn(out)
        # Force initial value of the horizon to be lower bounded by the final value of the input sequence.
        # Assumes the last feature to be the count of seismic events.
        # out = out.cumsum(-1) + x[:, -1, -1]
        return out
