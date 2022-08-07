import os
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from saif.ml_utils.data_utils import daily_seismic_and_interpolated_pressure
from saif.scinet.dataset import construct_time_series_dataset
import saif.scinet.model as scinet

import wandb

_FUNCS = {
    'abs' : lambda x : x.abs(),
    'quad' : lambda x : x ** 2,
    'relu' : F.relu,
    'exp' : torch.exp,
    'sigmoid' : torch.sigmoid,
    'no_func' : lambda x : x
}

class SimpleSCINet(nn.Module):
    def __init__(
        self, 
        input_len, output_len,
        input_dim, num_levels, kernel_size, dropout, groups, hidden_size,
        monotonic_fn=lambda x : x
    ):
        super().__init__()
        
        self.input_len = input_len
        self.output_len = output_len
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_levels = num_levels
        self.groups = groups
        self.kernel_size = kernel_size
        self.dropout = dropout
        
        self.bn1 = nn.BatchNorm1d(self.input_dim)
        self.bn2 = nn.BatchNorm1d(self.input_dim)
        
        self.block1 = scinet.EncoderTree(
            in_planes=self.input_dim,
            num_levels=self.num_levels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            groups=self.groups,
            hidden_size=self.hidden_size,
            INN=True
        )
        
        # I'm a bit iffy on using a projector like this across features
        # But this is what they do in scinet
        # It should be fine, kernel size is 1, it's essentially just an
        # aggregation operation
        self.time_projector = nn.Conv1d(
            self.input_len, self.output_len,
            kernel_size=1, stride=1, bias=False
        )
        
        self.channel_projector = nn.Conv1d(
            self.input_dim, 1, kernel_size=1, stride=1, bias=True
        )

        self.monotonic_fn = monotonic_fn
    
    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.bn1(out)
        out = out.permute(0, 2, 1)
        
        out = self.block1(out)
        out += out
        
        out = F.relu(out)
        out = self.time_projector(out)
        
        out = out.permute(0, 2, 1)
        out = self.bn2(out)
        out = F.relu(out)
        
        out = self.channel_projector(out).squeeze(1)
        
        #Enforce monotonicity
        out = self.monotonic_fn(out)

        out = out.cumsum(-1) + x[:, -1, -1, None]
        
        return out

def load_data(config):
    datapath = config.datapath
    
    seismic = pd.read_csv(os.path.join(datapath, 'seismic.csv'))
    pressure = pd.read_csv(os.path.join(datapath, 'pressure.csv'))

    # features, target_vals = daily_seismic_and_interpolated_pressure(seismic, pressure)
    features, t0 = daily_seismic_and_interpolated_pressure(seismic, pressure)
    target_vals = features.target

    if config.feature_set == 'full':
        feature_names = features.columns
    elif config.feature_set == 'injection':
        feature_names = ['pressure','dpdt','seismic']
    else:
        feature_names = ['seismic']

    train_dset, test_dset, _, _ = construct_time_series_dataset(
        features, target_vals, 
        config.input_len, config.horizon, feature_names, 
        train_test_split=config.train_test_split, normalize_data=True
    )

    train_loader = DataLoader(
        train_dset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=1
    )

    test_loader = DataLoader(
        test_dset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=1
    )

    input_dim = train_dset.X.shape[1]

    return input_dim, train_loader, test_loader

def build_model(input_dim, config):
    _func = _FUNCS.get(config.monotonic_activation)

    return SimpleSCINet(
        input_len=config.input_len, output_len=config.horizon,
        input_dim=input_dim, num_levels=config.num_levels, 
        kernel_size=config.kernel_size, dropout=config.dropout, 
        groups=1, hidden_size=config.hidden_size,
        monotonic_fn=_func
    )

def train_step(device, criterion, model, optimizer, train_loader):
    model.train()
    loss_vals = []
    for (batch_x, batch_y) in train_loader:
        optimizer.zero_grad()
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        loss_vals.append(loss.item())

    return np.mean(loss_vals)

def test_step(device, criterion, model, test_loader):
    model.eval()
    loss_vals = []
    for (batch_x, batch_y) in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss_vals.append(loss.item())

    return np.mean(loss_vals)

def run_exp(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        
        torch.manual_seed(config.seed)
        input_dim, train_loader, test_loader = load_data(config)
        model = build_model(input_dim, config)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.HuberLoss()

        wandb.watch(model)
        for epoch in range(config.n_epoch):
            train_loss = train_step(device, criterion, model, optimizer, train_loader)
            test_loss = test_step(device, criterion, model, test_loader)
            
            wandb.log({
                "epoch": epoch, 
                'train_loss' : train_loss,
                "test_loss": test_loss
            })  


def default_val(name, vals):
    if isinstance(vals, list):
        return {name : {'values' : vals}}
    else:
        return {name : {'value' : vals}}

def make_param_dict():
    # Hyperparameters
    parameters_dict = {}

    parameters_dict.update(
        default_val('seed', 0)
    )
    parameters_dict.update(
        # default_val('datapath', "gs://us-geomechanicsforco2-dev-staging/temporal_datasets/kansas/loc1")
        default_val('datapath','../../../dataset_preparing/Temporal_Datasets/kansas/loc1')
    )
    parameters_dict.update(
        default_val('feature_set', ['full', 'injection', 'seismic'])
    )
    parameters_dict.update(
        default_val('input_len', [16,32, 64])
    )
    parameters_dict.update(
        default_val('horizon', 7)
    )
    parameters_dict.update(
        default_val('train_test_split', 0.8)
    )
    parameters_dict.update(
        default_val('batch_size', [32, 64])
    )
    parameters_dict.update(
        default_val('num_levels', [1, 2, 3])
    )
    parameters_dict.update(
        default_val('kernel_size', [3, 4, 5])
    )
    parameters_dict.update(
        default_val('dropout', [0.2, 0.3, 0.4, 0.5])
    )
    parameters_dict.update(
        default_val('hidden_size', [1, 2, 3])
    )
    parameters_dict.update(
        default_val('monotonic_activation', _FUNCS['quad'])
    )
    parameters_dict.update(
        default_val('lr', 1e-3)
    )
    parameters_dict.update(
        default_val('n_epoch', 128)
    )

    return parameters_dict

def make_config():
    sweep_config = {
        'method': 'bayes'
    }

    metric = {
        'name': 'test_loss',
        'goal': 'minimize'
    }

    sweep_config['metric'] = metric

    sweep_config['parameters'] = make_param_dict()

    return sweep_config


if __name__ == "__main__":
    wandb.login()
    sweep_config = make_config()
    sweep_id = wandb.sweep(
        sweep_config, 
        project="scinet-kansas-1", 
        entity="fdl2022_team_geomechanics-for-co2-sequestration"
    )
    print("sweep_id:", sweep_id)
    # wandb.agent(sweep_id, function=run_exp, count=100)
    wandb.agent(sweep_id, function=run_exp)
