import torch, math
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from saif.scinet.dataset import TimeSeriesDataset
from saif.ml_utils.normalization import normalize
from typing import List, Tuple
####################################################################
class LSTMdataset(Dataset):
    def __init__(self, feature_data: np.ndarray, target_data: np.ndarray, seq_length: int) -> None:
        '''
        Parameters:
        ------------------
        feature_data: np.ndarray
            Feature data of shape (n_values, n_features)

        target_data: np.ndarray
            Target data of shape (n_values, n_target)

        seq_length: int
             Sequence length
        '''
        self.horizon = 1
        self.seq_length = seq_length
        self.X = torch.tensor(feature_data).float()
        self.Y = torch.tensor(target_data).float()

    def __len__(self):
        '''
        Checks for length of the dataset
        '''
        return self.X.shape[0]

    def __getitem__(self, i):
        '''
        Returns rows (i - seq_leng) to i upon querying i^th element of the dataset.
        If i is near the beginning of the dataset, we pad by repeating the first row as many times as needed to make the output have seq_length rows.
        '''
        if i >= self.seq_length - 1:
            x_start = i - self.seq_length + 1
            x_end = x_start + self.seq_length
            y_start = x_end
            y_end = y_start + self.horizon
            x = self.X[x_start:x_end, :]
            y = self.Y[y_start:y_end]
        else:
            padding = self.X[0].repeat(self.seq_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)
            y_start = i+1
            y = self.Y[i+1: i+self.horizon+1]
        return x, y

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

    # Generate data sets. Set horizon length  to 1.
    train_dset = TimeSeriesDataset(train_x, train_y, seq_length, 1, feature_names)
    test_dset = TimeSeriesDataset(test_x, test_y, seq_length, 1, feature_names)

    if do_normalize:
        return train_dset, test_dset, x_scaler, y_scaler
    else:
        return train_dset, test_dset
####################################################################
