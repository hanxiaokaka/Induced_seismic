import os
import json
import sys
import traceback

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from saif.ml_utils.data_utils import daily_seismic_and_interpolated_pressure
from saif.lstm.pl_lstm import SeismicDataModule, RNNForecaster
from saif.ml_utils.activations import MONOTONIC_FUNCS

import wandb

def load_data(config):
    datapath = config.datapath
    
    seismic = pd.read_csv(os.path.join(datapath, 'seismic.csv'), index_col=0)
    pressure = pd.read_csv(os.path.join(datapath, 'pressure.csv'), index_col=0)

    #extract features
    features, t0, target_vals = daily_seismic_and_interpolated_pressure(seismic, pressure)
    features['cum_counts'] = target_vals
    len_series = len(features)
    n_train = int(config.train_frac * len_series)
    n_val = int(config.val_frac * len_series)

    #scale features
    scaler = MinMaxScaler().fit(features[:n_train])
    scaled_features = scaler.transform(features)
    scaled_features = pd.DataFrame(scaled_features, columns=features.columns)
    scaled_features['index'] = np.arange(len(features))
    scaled_features['scaled_index'] = scaled_features['index'] / len(features)

    if config.feature_set == 'full':
        feature_names = ['pressure', 'dpdt', 'scaled_index']
    else:
        feature_names = ['scaled_index']

    Y = torch.FloatTensor(scaled_features.cum_counts.values)
    X = torch.FloatTensor(scaled_features[feature_names].values)

    Yt, Yv, Ytest = Y[None, :n_train, None], Y[None, :n_train + n_val, None], Y[None, :, None]
    Xt, Xv, Xtest = X[None, :n_train, :], X[None, :n_train + n_val, :], X[None, :, :]

    data_module = SeismicDataModule(Xt, Yt, Xv, Yv, Xtest, Ytest)

    return data_module

# self, 
# input_size, 
# hidden_size, 
# num_layers, 
# dropout, 
# learning_rate,
# criterion,
# monotonic_fn='sqrt',
# use_bias=False,
# rnn_type='rnn'
# ):
def build_model(input_size, config):
    return RNNForecaster(
        input_size=input_size, hidden_size=config.hidden_size, 
        num_layers=config.num_layers, dropout=config.dropout, 
        learning_rate=config.learning_rate, criterion=F.mse_loss,
        monotonic_fn=config.monotonic_fn, use_bias=False,
        rnn_type=config.rnn_type
    )

def run_exp(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        seed_everything(config.seed)

        try:
            data_module = load_data(config)
            input_dim = data_module.Xt.shape[-1]
            model = build_model(input_dim, config)

            wandb_logger = WandbLogger(log_model='all')
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss", mode="min", dirpath=wandb.run.dir
            )

            trainer = Trainer(
                max_epochs=config.max_epochs,
                logger=wandb_logger,
                log_every_n_steps=1,
                accelerator='gpu', 
                devices=1,
                callbacks=[checkpoint_callback],
                enable_checkpointing=True
            )

            trainer.fit(model, data_module)
            trainer.test(model, data_module)
        except Exception as e:
            # exit gracefully, so wandb logs the problem
            print(traceback.print_exc(), file=sys.stderr)
            exit(1)


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
        default_val('datapath', "/home/caesar/saif/data/02_intermediate")
    )
    parameters_dict.update(
        default_val('feature_set', ['full', 'injection'])
    )
    parameters_dict.update(
        default_val('train_test_split', 0.7)
    )
    parameters_dict.update(
        default_val('dropout', [0.1, 0.2, 0.3, 0.4, 0.5])
    )
    parameters_dict.update(
        default_val('num_layers', [1, 2, 3])
    )
    parameters_dict.update(
        default_val('rnn_type', ['rnn', 'lstm'])
    )
    parameters_dict.update(
        default_val('monotonic_fn', list(MONOTONIC_FUNCS.keys()))
    )
    parameters_dict.update(
        default_val('max_epochs', 500)
    )
    parameters_dict.update(
        default_val('train_frac', 0.7)
    )
    parameters_dict.update(
        default_val('val_frac', 0.2)
    )

    return parameters_dict

def make_config():
    sweep_config = {
        'method': 'bayes'
    }

    metric = {
        'name': 'val_loss',
        'goal': 'minimize'
    }

    sweep_config['metric'] = metric

    sweep_config['parameters'] = make_param_dict()

    #distributions
    sweep_config['parameters'].update({
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-2
        },
        'hidden_size': {
            'distribution': 'q_uniform',
            'q' : 1,
            'min': 1,
            'max': 512,
        }
    })

    return sweep_config


if __name__ == "__main__":
    wandb.login()
    sweep_config = make_config()
    sweep_id = wandb.sweep(
        sweep_config, 
        project="lstm-final-report", 
        entity="fdl2022_team_geomechanics-for-co2-sequestration"
    )
    print("sweep_id:", sweep_id)
    wandb.agent(sweep_id, function=run_exp, count=50)
    
