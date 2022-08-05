import torch, math
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from saif.scinet.dataset import TimeSeriesDataset
from saif.ml_utils.normalization import normalize
from typing import List, Tuple
####################################################################
def construct_dataset(features: pd.DataFrame, target_vals: np.ndarray, seq_length: int,
                      feature_names: List[str], train_frac: float = 0.7,
                      do_normalize: bool = True):
    '''
    Perform train-test split, data normalization, and return transformed feature and target data.

    Parameters:
    -------------
    features: pd.DataFrame
        Feature data

    target_vals: 1D NumPy array
        Target data

    seq_length: int
        Input or sequence length to machine learning model

    feature_names: list of strings
        List of features of interest for modeling

    train_frac: float
        Fraction of full data to be labeled as training data. The remainder gets marked as test data.

    do_normalize: bool
        Flag that dictates if data normalization is to be performed

    '''
    # Size of training data
    n_train = math.floor(train_frac * len(features))
    # Train-test split
    train_x, test_x = features[feature_names].iloc[:n_train], features[feature_names].iloc[n_train:]
    train_y, test_y = target_vals[:n_train], target_vals[n_train:]

    # Data normalization
    if do_normalize:
        train_x, train_y, test_x, test_y, x_scaler, y_scaler = normalize(train_x, train_y, test_x, test_y)

    # Generate data sets. Set horizon length to 1.
    train_dset = TimeSeriesDataset(train_x, train_y, seq_length, 1, feature_names)
    test_dset = TimeSeriesDataset(test_x, test_y, seq_length, 1, feature_names)

    if do_normalize:
        return train_dset, test_dset, x_scaler, y_scaler
    else:
        return train_dset, test_dset
####################################################################
