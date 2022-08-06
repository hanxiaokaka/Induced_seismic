import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from typing import Tuple
####################################################################
def normalize(train_x: pd.DataFrame, train_y: np.ndarray, val_x: pd.DataFrame, val_y: np.ndarray, test_x: pd.DataFrame, test_y: np.ndarray
             ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, RobustScaler, RobustScaler]:
    '''
    Standardize training and test data using the RobustScaler method of scikit-learn.
    This Scaler removes the median and scales the data according to the quantile range.

    Parameters:
    ----------------------
    train_x: {array-like, sparse matrix} of shape (n_samples_train, n_features)
    train_y: {array-like, sparse matrix} of shape (n_samples_train, )
    val_x: {array-like, sparse matrix} of shape (n_samples_val, n_features)
    val_y: {array-like, sparse matrix} of shape (n_samples_val, )
    test_x: {array-like, sparse matrix} of shape (n_samples_test, n_features)
    test_y: {array-like, sparse matrix} of shape (n_samples_test, )

    Returns:
    ----------------------
    sccaled_train_x: {array-like, sparse matrix} of shape (n_samples_train, n_features)
    scaled_train_y: {array-like, sparse matrix} of shape (n_samples_train, )
    scaled_val_x: {array-like, sparse matrix} of shape (n_samples_val, n_features)
    scaled_val_y: {array-like, sparse matrix} of shape (n_samples_val, )    
    scaled_test_x: {array-like, sparse matrix} of shape (n_samples_test, n_features)
    scaled_test_y: {array-like, sparse matrix} of shape (n_samples_test, )
    x_scaler: RobustScaler
    y_scaler: RobustScaler
    '''
    x_scaler, y_scaler = RobustScaler(), RobustScaler()

    scaled_train_x = x_scaler.fit_transform(train_x)
    scaled_val_x = x_scaler.transform(val_x)
    scaled_test_x = x_scaler.transform(test_x)

    scaled_train_y = y_scaler.fit_transform(train_y[:, None]).squeeze(1)
    scaled_val_y = y_scaler.transform(val_y[:, None]).squeeze(1)
    scaled_test_y = y_scaler.transform(test_y[:, None]).squeeze(1)

    return scaled_train_x, scaled_train_y, scaled_val_x, scaled_val_y, scaled_test_x, scaled_test_y, x_scaler, y_scaler
####################################################################
