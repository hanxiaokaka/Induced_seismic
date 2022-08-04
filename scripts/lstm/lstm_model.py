import os, time
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from saif.ml_utils.data_utils import daily_seismic_and_interpolated_pressure
from saif.lstm.dataset import construct_dataset
from saif.lstm.model import ShallowRegLSTM
from saif.lstm.train_utils import train_model, test_model, predict
from saif.lstm.plot_utils import plot_losscurve, plot_modelpred
####################################################################
# INPUTS

config = {
# Location tag for plots and directory labels
'location': 'cushing2014',
# Seismic data
'seismic_csv': 'gs://us-geomechanicsforco2-dev-staging/temporal_datasets/cushing_2014_oklahoma/seismic.csv',
# Pressure data
'pressure_csv': 'gs://us-geomechanicsforco2-dev-staging/temporal_datasets/cushing_2014_oklahoma/pressure.csv',
# Features of interest
'feature_names': ['pressure', 'dpdt'],
# Fraction of full data kept aside as training data
'train_frac': 0.8,
# Sequence length
'seq_length': 16,
# Batch size
'batch_size': 8,
# Hidden size of LSTM
'hidden_size': 16,
# Number of recurrent layers in LSTM
'num_layers': 1,
# Dropout probability
'dropout': 0.0,
# Monotonic activation function
'monotonic_fn': lambda x: x.abs(),
# Learning rate
'lr': 1.0e-5,
# Loss criterion
'criterion': 'Huber', # Label for plotting
'loss_function': nn.HuberLoss(),
# Max no. of epochs of training
'max_epoch': 500,
# Path to save model parameters at each epoch of training
'PARAMS_DIR': '../../data/06_models/lstm/',
# Path to plots
'PLOT_DIR': '../../plots/lstm/',
# Output plot formats
'plot_formats': ['.png']
}
####################################################################
def fit_lstm(config: dict) -> None:
    # Read in data.
    seismic_df = pd.read_csv(config['seismic_csv'])
    pressure_df = pd.read_csv(config['pressure_csv'])
    # Update paths.
    config['PARAMS_DIR'] += '%s_train%s/'% (config['location'], config['train_frac'])
    config['PLOT_DIR'] += '%s_train%s/'% (config['location'], config['train_frac'])

    # Create output directories if non-existent.
    if not os.path.isdir(config['PARAMS_DIR']):
        os.makedirs(config['PARAMS_DIR'])
    if not os.path.isdir(config['PLOT_DIR']):
        os.makedirs(config['PLOT_DIR'])

    features, t0, target_vals = daily_seismic_and_interpolated_pressure(seismic_df, pressure_df)
    print('Aggregated pressure and seismic data.')

    train_dset, test_dset, x_scaler, y_scaler = construct_dataset(features, target_vals, config['seq_length'],
                                                                  config['feature_names'], config['train_frac'],
                                                                  do_normalize=True)
    print('Train/test data set generation completed.')

    # Set up data loaders.
    train_loader = DataLoader(train_dset, config['batch_size'], shuffle=True)
    test_loader =  DataLoader(test_dset, config['batch_size'])

    # Intialize model.
    model = ShallowRegLSTM(len(config['feature_names']), config['hidden_size'], config['num_layers'],
                           config['dropout'], config['monotonic_fn'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    print('LSTM model created and optimizer initialized.')

    # Model training
    start_time = time.time()
    n_epoch = np.arange(config['max_epoch'])+1
    train_loss = np.zeros(config['max_epoch'])
    test_loss = np.zeros(config['max_epoch'])
    print('Beginning training')
    for idx_epoch in range(config['max_epoch']):
        print('EPOCH %d -------'% (idx_epoch+1))
        train_loss[idx_epoch] = train_model(train_loader, model, config['loss_function'], optimizer)
        test_loss[idx_epoch] = test_model(test_loader, model, config['loss_function'])
        # Save model parameters to disk for every epoch of traning.
        torch.save(model.state_dict(), config['PARAMS_DIR']+'epoch%d.h5'% (idx_epoch+1))
        print()
    end_time = time.time()
    train_time = (end_time - start_time)/60.0 # Training time (minutes)
    print('Model training took %.2f minutes.'% (train_time))

    # Find epoch of robust fit.
    idx_robustfit = np.argmin(test_loss)
    print('Loss on test data was minimized at epoch %d.'% (idx_robustfit+1))
    # Load parameter values from epoch of robust fit.
    model.load_state_dict(torch.load(config['PARAMS_DIR']+'epoch%d.h5'% (idx_robustfit+1)))

    # Obtain model prediction on test data.
    train_eval_loader = DataLoader(train_dset, batch_size=config['batch_size'], shuffle=False)
    prediction_train = predict(train_eval_loader, model)
    prediction_test = predict(test_loader, model)
    # Undo normalization.
    y_train = y_scaler.inverse_transform(train_dset.y.numpy().reshape(-1,1)).flatten()
    y_test = y_scaler.inverse_transform(test_dset.y.numpy().reshape(-1,1)).flatten()
    prediction_train = y_scaler.inverse_transform(prediction_train.numpy().reshape(-1,1)).flatten()
    prediction_test =  y_scaler.inverse_transform(prediction_test.numpy().reshape(-1,1)).flatten()
    print('Computed model prediction on training and test data')

    print('Plotting loss curve')
    plot_losscurve(n_epoch, train_loss, test_loss, config['criterion'], config['PLOT_DIR']+'losscurve', config['plot_formats'])

    print('Plotting train/test data and model prediction')
    test_start_idx = len(train_dset.y)
    plot_modelpred(features.days[:test_start_idx], y_train, features.days[test_start_idx:], y_test,
                   prediction_train, prediction_test, t0, config['PLOT_DIR']+'prediction', config['plot_formats'])

    # Save loss curve data to file.
    losscurve_df = pd.DataFrame({'epoch':n_epoch, 'train_loss':train_loss, 'test_loss':test_loss})
    losscurve_df.to_csv(config['PARAMS_DIR']+'losscurve.csv',index=None)
    print('Loss curve data written to file.')

    # Save model predictions on training data to file.
    pred_test_df = pd.DataFrame({'days': features.days[:test_start_idx].values, 'y_train': y_train,
                                 'prediction': prediction_train})
    pred_test_df.to_csv(config['PARAMS_DIR']+'train_pred.csv', index=None)
    print('Model predictions on training data written to file.')

    # Save model predictions on test data to file.
    pred_test_df = pd.DataFrame({'days': features.days[test_start_idx:].values, 'y_test': y_test,
                                 'prediction':prediction_test})
    pred_test_df.to_csv(config['PARAMS_DIR']+'/test_pred.csv', index=None)
    print('Model predictions on test data written to file.')

####################################################################
if __name__ == "__main__":
    fit_lstm(config)
####################################################################
