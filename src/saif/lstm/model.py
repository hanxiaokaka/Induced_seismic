from torch import nn
import torch
import numpy as np

class ShallowRegLSTM(nn.Module):
    def __init__(self, N_features, N_hidden):
        '''
        Shallow LSTM model for regression

        Parameters:
        -------------
        N_features:  (int) Number of input features
        N_hidden: (int) Number of hidden units in LSTM
        '''
        super(ShallowRegLSTM, self).__init__()

        self.input_size = N_features
        self.hidden_units = N_hidden
        self.num_layers = 1 #  Only 1 output layer for regression task

        # By default, nn.LSTM() assumes the batch number to be the second index.
        # We use batch_first=True to conform with our data set structure.
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_units,
                            batch_first=True, num_layers=self.num_layers)
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        '''
        Forward pass of input data through the model
        '''
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        # First dimension of hn is num_layers, which is set to 1 here for regression.
        out = self.linear(hn[0]).flatten()
        # Squaring to get a positive output
        #out = out**2
        # Enforce monotonicity.
        #out = out.cumsum(-1)
        return out
