import os

import numpy as np
import pandas as pd

from saif.ml_utils.data_utils import daily_seismic_and_interpolated_pressure
from saif.scinet.dataset import construct_time_series_dataset

from lightgbm import LGBMRegressor

import wandb

from wandb.lightgbm import wandb_callback, log_summary
from sklearn.metrics import mean_squared_error

import datetime as dt


def extract_arrays(dset):
    X = []
    Y = []
    for i in range(len(dset)):
        _x, _y = dset[i]
        X.append(_x.flatten().numpy())
        Y.append(_y.numpy())
    X, Y = np.array(X), np.array(Y).squeeze(-1)
    return X, Y

def load_data(config):
    datapath = config.datapath
    
    seismic = pd.read_csv(os.path.join(datapath, 'seismic.csv'), index_col=0)
    pressure = pd.read_csv(os.path.join(datapath, 'pressure.csv'), index_col=0)

    features, _, target_vals = daily_seismic_and_interpolated_pressure(
        seismic, pressure, target_fn=lambda x : x
    )

    if config.feature_set == 'full':
        feature_names = features.columns
    elif config.feature_set == 'injection':
        feature_names = ['pressure','dpdt','counts']
    else:
        feature_names = ['counts']

    features = features[feature_names]
    train_dset, test_dset = construct_time_series_dataset(
        features, target_vals, 
        config.input_len, config.horizon, feature_names, 
        train_test_split=config.train_test_split, normalize_data=False
    )

    Xt, Yt = extract_arrays(train_dset)
    Xv, Yv = extract_arrays(test_dset)

    # normalize by std
    # This is a placeholder for now...we cannot use the usual normalizer
    # Because this can produce negative values
    # Which we cannot use with poisson regression

    #The scaling needs to be done so we can still compute mse of the forecast
    scale = Yt.std()
    Yt = Yt / scale
    Yv = Yv / scale

    return Xt, Yt, Xv, Yv

def build_model(config):
    return LGBMRegressor(
        objective='poisson',
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        boosting=config.boosting,
        learning_rate=config.learning_rate,
        seed=config.seed
    )

def run_exp(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        Xt, Yt, Xv, Yv = load_data(config)
        model = build_model(config)
        model.fit(Xt, Yt, callbacks=[wandb_callback(log_params=False)])
        #You must pass the booster to log_summary
        #Uncomment to generate feature importance plots, and save models
        #log_summary(model.booster_, save_model_checkpoint=True)

        #Evaluate and log metric for tuning
        predicted = model.predict(Xv)
    
        samples = np.arange(len(Yv))
        sum_true = np.cumsum(Yv)
        sum_pred = np.cumsum(predicted)

        wandb.log({'mse' : mean_squared_error(sum_true, sum_pred)})

        #Generate additional plots
        unique_str_id = "line-series-plot-{}".format(dt.datetime.now())
        wandb.log({
            unique_str_id: wandb.plot.line_series(
                samples.tolist(), 
                [sum_pred.tolist(), sum_true.tolist()], 
                title="Forecasts", 
                keys=["pred", "true"],
                xname="=step")
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
        default_val('datapath', "/home/caesar/saif/data/02_intermediate")
    )
    parameters_dict.update(
        default_val('feature_set', ['full', 'injection', 'seismic'])
    )
    parameters_dict.update(
        default_val('horizon', 1)
    )
    parameters_dict.update(
        default_val('train_test_split', 0.8)
    )
    parameters_dict.update(
        default_val('boosting', ['gbdt', 'dart', 'goss'])
    )

    return parameters_dict

def make_config():
    sweep_config = {
        'method': 'bayes'
    }

    metric = {
        'name': 'mse',
        'goal': 'minimize'
    }

    sweep_config['metric'] = metric

    sweep_config['parameters'] = make_param_dict()

    #distributions
    sweep_config['parameters'].update({
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e1
        },
        'max_depth': {
            'distribution': 'q_log_uniform_values',
            'q': 1,
            'min': 2,
            'max': 128,
        },
        'input_len': {
            'distribution': 'q_log_uniform_values',
            'q': 1,
            'min': 1,
            'max': 64,
        },
        'n_estimators': {
            'distribution': 'q_log_uniform_values',
            'q': 1,
            'min': 2 ** 3,
            'max': 2 ** 14,
        }
    })


    return sweep_config


if __name__ == "__main__":
    wandb.login()
    sweep_config = make_config()
    sweep_id = wandb.sweep(
        sweep_config, 
        project="gb_tree-demo-v1", 
        entity="fdl2022_team_geomechanics-for-co2-sequestration"
    )
    print("sweep_id:", sweep_id)
    wandb.agent(sweep_id, function=run_exp, count=200)
