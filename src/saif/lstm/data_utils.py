import numpy as np
import pandas as pd

def bin_data(seismic_df, N_samples=100):
    '''
    Bin seismic data into a timeseries of length N_samples. Then, perform train-test split.

    Parameters:
    -------------
    seismic_df: pd.DataFrame
    N_samples: (int) Number of time samples desired across binned seismic data
    horizon_len: (int) Horizon length of forecast

    Returns:
    -------------
    counts_df: pd.DataFrame of earthquake counts
    test_df: pd.DataFrame of test data
    '''
    t_binedges= np.linspace(seismic_df['epoch'].min(), seismic_df['epoch'].max(), N_samples+1)
    t_bincenters = 0.5*(t_binedges[1:] + t_binedges[:-1])
    # Bin earthquake counts into timeseries of length N_samples.
    idx_quakes = np.digitize(seismic_df['epoch'], t_binedges, right=False)
    counts_per_bin = np.array([np.size(np.where(idx_quakes==n+1)) for n in range(N_samples)])
    # Set up dataframe of earthquake counts.
    counts_df = pd.DataFrame({'epoch':t_bincenters, 'counts':counts_per_bin, 'cum_counts': np.cumsum(counts_per_bin)})
    return counts_df

def train_test_split(df, horizon_len=20):
    '''
    Train/test split a pandas DataFrame object for time series forecast.

    Parameters:
    -------------
    df: pd.DataFrame
    horizon_len: (int) Forecast horizon length in number of samples

    Returns:
    -------------
    train_df: pd.DataFrame of training data
    test_df: pd.DataFrame of test data
    '''
    test_start_idx = len(df) - horizon_len
    if test_start_idx<=0:
        print('ERROR: Specified horizon exceeds length of data set. Train-test split not possible.')
        # Return dummy values.
        return 0, 0
    else:
        train_df = df.iloc[:test_start_idx].copy()
        test_df = df.iloc[test_start_idx:].copy()
        return train_df, test_df

def standardize(train_df, test_df, target):
    '''
    Standardize both the features and the target using their respective means and standard deviation from the training data.

    Parameters:
    -------------
    train_df: pd.DataFrame of training data
    test_df: pd.DataFrame of test data
    target: (str) Target column in training and test dataframes

    Returns:
    -------------
    train_df: pd.DataFrame of normalized training data
    test_df: pd.DataFrame of normalized test data
    target_mean: (float) Mean value of target column in training data
    target_stdev: (float) Standard deviation of target column in training data
    '''
    target_mean = train_df[target].mean()
    target_stdev = train_df[target].std()

    train_df_normed = train_df.copy()
    test_df_normed = test_df.copy()

    for col in train_df.columns:
        mean = train_df[col].mean()
        stdev = train_df[col].std()

        train_df_normed[col] = (train_df_normed[col] - mean)/stdev
        test_df_normed[col] = (test_df_normed[col] - mean)/stdev

    return train_df_normed, test_df_normed, target_mean, target_stdev
