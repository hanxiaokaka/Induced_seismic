# Hyperparameter tuning for LSTM model
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
from saif.lstm.dataset import construct_dataset
from saif.lstm.model import ShallowRegLSTM
from saif.lstm.train_utils import train_model, test_model, predict
from saif.lstm.plot_utils import plot_losscurve, plot_modelpred
import wandb

# Monotonic activation functions
_FUNCS = {
    'abs' : lambda x : x.abs(),
    'quad' : lambda x : x ** 2,
    'relu' : F.relu,
    'exp' : torch.exp,
    'sigmoid' : torch.sigmoid,
    'identity' : lambda x : x
}

def load_data(config):
    datapath = config.datapath
    seismic = pd.read_csv(os.path.join(datapath, 'seismic.csv'))
    pressure = pd.read_csv(os.path.join(datapath, 'pressure.csv'))
    features, t0, target_vals = daily_seismic_and_interpolated_pressure(seismic, pressure)
    features['seismic'] = target_vals

    if config.feature_set == 'full':
        feature_names = features.columns
    elif config.feature_set == 'injection':
        feature_names = ['pressure','dpdt','seismic']
    else:
        feature_names = ['seismic']
    N_features = len(feature_names)
    train_dset, test_dset, x_scaler, y_scaler = construct_dataset(features, target_vals, config.input_len, feature_names,
                                                                  config.train_test_split, do_normalize=True)
    # Initialize data loaders.
    train_loader = DataLoader(train_dset, config.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dset, config.batch_size, shuffle=False, num_workers=1)

    return N_features, train_loader, test_loader

def build_model(N_features, config):
    _func = _FUNCS.get(config.monotonic_activation)
    return ShallowRegLSTM(N_features, config.hidden_size, config.num_layers, config.dropout, _func)

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
        N_features, train_loader, test_loader = load_data(config)
        model = build_model(N_features, config)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.HuberLoss()

        wandb.watch(model)
        for epoch in range(config.n_epoch):
            train_loss = train_step(device, criterion, model, optimizer, train_loader)
            test_loss = test_step(device, criterion, model, test_loader)
            wandb.log({"epoch": epoch, 'train_loss' : train_loss, "test_loss": test_loss})

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
        default_val('datapath', "gs://us-geomechanicsforco2-dev-staging/temporal_datasets/cushing_2014_oklahoma")
    )
    parameters_dict.update(
        default_val('feature_set', ['full', 'injection', 'seismic'])
    )
    parameters_dict.update(
        default_val('input_len', [4, 8, 16, 32, 64])
    )
    parameters_dict.update(
        default_val('train_test_split', 0.8)
    )
    parameters_dict.update(
        default_val('batch_size', [16, 32, 64])
    )
    parameters_dict.update(
        default_val('num_layers', [1, 2, 3])
    )
    parameters_dict.update(
        default_val('dropout', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    )
    parameters_dict.update(
        default_val('hidden_size', [4, 8, 16, 32, 64])
    )
    parameters_dict.update(
        default_val('monotonic_activation', list(_FUNCS.keys()))
    )
    parameters_dict.update(
        default_val('lr', [1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
    )
    parameters_dict.update(
        default_val('n_epoch', 512)
    )
    return parameters_dict

def make_config():
    sweep_config = {'method': 'bayes'}
    metric = {'name': 'test_loss',
              'goal': 'minimize'}
    sweep_config['metric'] = metric
    sweep_config['parameters'] = make_param_dict()
    return sweep_config

if __name__ == "__main__":
    wandb.login()
    sweep_config = make_config()
    sweep_id = wandb.sweep(
        sweep_config,
        project="lstm-tuning-v1",
        entity="fdl2022_team_geomechanics-for-co2-sequestration"
    )
    print("sweep_id:", sweep_id)
    wandb.agent(sweep_id, function=run_exp, count=100)
