import math

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from scipy.interpolate import interp1d
from sklearn.preprocessing import RobustScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, input_series, target_series, input_len, horizon, feature_names=None):
        self.X = torch.FloatTensor(input_series)
        self.Y = torch.FloatTensor(target_series)
        self.input_len = input_len
        self.horizon = horizon

        self.feature_names = feature_names

    def __getitem__(self, index):
        x_start = index
        x_end = x_start + self.input_len
        y_start = x_end + 1
        y_end = y_start + self.horizon
        
        return self.X[x_start:x_end], self.Y[y_start:y_end]
    
    def __len__(self):
        #TODO: double check off-by-one
        return len(self.Y) - self.input_len - self.horizon


def overlap_and_interpolate(seismic, pressure):
    ## Find overlap
    seismic_start = seismic.epoch.values[0]
    seismic_end = seismic.epoch.values[-1]
    
    pressure_start = pressure.epoch.values[0]
    pressure_end = pressure.epoch.values[-1]
    
    #start time
    t0 = max(seismic_start, pressure_start)
    #end time
    t1 = min(seismic_end, pressure_end)

    #convert epoch to days
    ep2day = 1. / 86400
    seismic['days'] = (seismic.epoch - t0) * ep2day
    pressure['days'] = (pressure.epoch - t0) * ep2day

    dt = (t1 - t0) * ep2day
    bins = np.arange(0, math.ceil(dt) + 1)

    seismic['t'] = np.digitize(
        seismic['days'], bins
    )

    # Interpolate pressure
    p_func = interp1d(
        pressure.days, pressure.pressure, 
        kind='linear'
    )
    dpdt_func = interp1d(
        pressure.days, pressure.dpdt, 
        kind='linear'
    )

    resampled_pressure_df = {}
    resampled_pressure_df['pressure'] = p_func(bins)
    resampled_pressure_df['dpdt'] = dpdt_func(bins)
    
    resampled_pressure_df = pd.DataFrame(resampled_pressure_df)

    return seismic, resampled_pressure_df, bins

def aggregate_seismic(seismic, n_steps, features, bin_name='days'):
    seismic_counts = seismic[[bin_name,'t']].groupby('t').agg('count')
    
    seismic_features = seismic[features].groupby('t').agg([np.mean, np.std]).fillna(0)
    seismic_features.columns = ['_'.join(col).strip() for col in seismic_features.columns.values]
    
    n_features = len(seismic_features.columns)
    
    output_vals = np.zeros((n_steps, n_features))
    output_vals[seismic_features.index.values] = seismic_features.values

    output_df = pd.DataFrame(output_vals, columns=seismic_features.columns)

    target_vals = np.zeros((n_steps,))
    target_vals[seismic_counts.index.values] = seismic_counts[bin_name].values
    
    return output_df, target_vals

def normalize(train_x, train_y, test_x, test_y):
    x_scaler, y_scaler = RobustScaler(), RobustScaler()

    train_x = x_scaler.fit_transform(train_x)
    test_x = x_scaler.transform(test_x)

    train_y = y_scaler.fit_transform(train_y[:, None]).squeeze(1)
    test_y = y_scaler.transform(test_y[:, None]).squeeze(1)

    return train_x, train_y, test_x, test_y, x_scaler, y_scaler

def construct_time_series_dataset(
        features, target_vals, 
        input_len, horizon, feature_names, 
        train_test_split=0.7, normalize_data=True
    ):
    features = features[feature_names] # potential fix
    n_train = math.floor(train_test_split * len(features))

    train_x, test_x = features.values[:n_train], features.values[n_train:]
    train_y, test_y = target_vals[:n_train], target_vals[n_train:]

    if normalize_data:
        train_x, train_y, test_x, test_y, x_scaler, y_scaler = normalize(
            train_x, train_y, test_x, test_y
        )

    train_dset = TimeSeriesDataset(
        train_x, train_y, input_len, horizon, feature_names
    )
    test_dset = TimeSeriesDataset(
        test_x, test_y, input_len, horizon, feature_names
    )


    if normalize_data:
        return train_dset, test_dset, x_scaler, y_scaler
    else:
        return train_dset, test_dset


def aggregate_seismic_between_pressure_readings(
        seismic, pressure, target_fn=np.cumsum
    ):
    _features = ['depth', 'easting', 'northing', 'magnitude', 't']
    bins = pressure.epoch.values
    n_steps = len(bins)

    # Aggregate the seismic events that occur between pressure readings
    seismic['t'] = np.digitize(
        seismic.epoch.values, 
        bins
    )
    
    seismic = seismic[seismic.epoch_bin < n_steps]

    seismic_features, target_vals = aggregate_seismic(
        seismic, n_steps, _features, bin_name='epoch'
    )
    target_vals = target_fn(target_vals)

    features = pd.concat([seismic_features, pressure], axis=1)
    features['seismic'] = target_vals

    return features, target_vals

def daily_seismic_and_interpolated_pressure(
        seismic, pressure, target_fn=np.cumsum
        #input_len=64, horizon=8, normalize_data=True, train_test_split=0.7
    ):
    #TODO: careful, overlap and interpolate does inplace modifications!
    _features = ['depth', 'easting', 'northing', 'magnitude', 't']
    seismic, pressure, bins = overlap_and_interpolate(seismic, pressure)
    seismic_features, target_vals = aggregate_seismic(seismic, len(bins), _features, bin_name='days')
    target_vals = target_fn(target_vals)

    features = pd.concat([seismic_features, pressure], axis=1)
    features['seismic'] = target_vals

    return features, target_vals

