import torch, math, sys
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from saif.scinet.dataset import TimeSeriesDataset
from saif.ml_utils.normalization import normalize
from typing import List, Tuple
####################################################################
def construct_dataset(features: pd.DataFrame, target_vals: np.ndarray, seq_length: int,
                      feature_names: List[str], train_frac: float = 0.64, val_frac: float = 0.16,
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
        Fraction of full data to be labeled as training data.

    val_frac: float
       Fraction of full data set as validation data. Any data that aren't marked as train or validation data get assigned as test data.

    do_normalize: bool
        Flag that dictates if data normalization is to be performed

    '''
    # Start index of validation data = Length of training data set
    val_start_idx = math.floor(train_frac * len(features))
    # Start index of test data = Start index of validation data + Length of validation data set
    test_start_idx = val_start_idx + math.floor(val_frac * len(features))

    # Train-val-test split
    train_x, train_y = features[feature_names].iloc[:val_start_idx], target_vals[:val_start_idx]
    val_x, val_y = features[feature_names].iloc[val_start_idx:test_start_idx], target_vals[val_start_idx:test_start_idx]
    test_x, test_y = features[feature_names].iloc[test_start_idx:], target_vals[test_start_idx:]

    # Data normalization
    if do_normalize:
        train_x, train_y, val_x, val_y, test_x, test_y, x_scaler, y_scaler = normalize(train_x, train_y, val_x, val_y, test_x, test_y)

    # Generate data sets. Set horizon length to 1.
    horizon_length = 1
    train_dset = TimeSeriesDataset(train_x, train_y, seq_length, horizon_length, feature_names)
    val_dset = TimeSeriesDataset(val_x, val_y, seq_length, horizon_length, feature_names)
    test_dset = TimeSeriesDataset(test_x, test_y, seq_length, horizon_length, feature_names)

    if do_normalize:
        return train_dset, val_dset, test_dset, x_scaler, y_scaler
    else:
        return train_dset, val_dset, test_dset
####################################################################
def concat_datasets(x1: torch.tensor, y1: torch.tensor, x2: torch.tensor, y2: torch.tensor, seq_length: int,
                    horizon_length: int, feature_names: List[str]):
    '''
    Concatenate two data sets. Assumes only a single target variable across input data sets.

    Parameters:
    -------------
    x1: torch.tensor of shape (N1, N_features)
        Features values in first data set. Here, N1 is the number of samples in the first data set.

    y1: torch.tensor of shape (N1,)
        Target data in first data set.

    x2: torch.tensor of shape (N2, N_features)
        Features values in the second data set. Here, N2 is the number of samples in the second data set.

    y2: torch.tensor of shape (N2, )
        Target data in the second data set.

    seq_length: int
        Input or sequence length to machine learning model

    horizon_length: int
       Forecast horizon length

    feature_names: list of strings
        List of features names

    Returns:
    -------------
    merged_dset: TimeSeriesDataset object
       Concatenated data set
    '''
    # Sanity checks
    if x1.shape[1]!=x2.shape[1]:
        print('The two data sets do not contain the same number of features. Merge infeasible.')
        sys.exit(1)
    if x1.shape[0]!=y1.shape[0]:
        print('x1 and y1 do not contain the sample number of samples. Terminating data set merge.')
        sys.exit(1)
    if x2.shape[0]!=y2.shape[0]:
        print('x2 and y2 do not contain the sample number of samples. Terminating data set merge.')
        sys.exit(1)

    x = torch.cat((x1, x2))
    y = torch.cat((y1, y2))
    merged_dset = TimeSeriesDataset(x, y, seq_length, horizon_length, feature_names)
    return merged_dset
####################################################################
