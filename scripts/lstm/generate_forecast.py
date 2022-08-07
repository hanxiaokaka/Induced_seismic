import os, time
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from saif.ml_utils.data_utils import daily_seismic_and_interpolated_pressure
from saif.lstm.dataset import construct_dataset, concat_datasets
from saif.lstm.model import ShallowRegLSTM
from saif.lstm.train_utils import train_model, test_model, unroll_forecast
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
# Independent variables of interest
'feature_names': ['pressure', 'dpdt'],
# PyTorch manual seed
'seed': 0,
# Fraction of full data kept aside as training data
'train_frac': 0.64,
# Fraction of full data set aside as validation data
'val_frac': 0.195,
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
'max_epoch': 1000,
# Path to save model parameters at each epoch of training
'PARAMS_DIR': '../../data/06_models/lstm/',
# Path to plots
'PLOT_DIR': '../../plots/lstm/',
# Output plot formats
'plot_formats': ['.png']
}
####################################################################
def fit_lstm(config: dict) -> None:
    torch.manual_seed(config['seed'])
    config['horizon_length'] = 1
    # Read in data.
    seismic_df = pd.read_csv(config['seismic_csv'])
    pressure_df = pd.read_csv(config['pressure_csv'])
    # Update paths.
    config['PARAMS_DIR'] += '%s_train%.1f_val%.1f/'% (config['location'], config['train_frac']*100, config['val_frac']*100)
    config['PLOT_DIR'] += '%s_train%.1f_val%.1f/'% (config['location'], config['train_frac']*100, config['val_frac']*100)

    # Create output directories if non-existent.
    if not os.path.isdir(config['PARAMS_DIR']):
        os.makedirs(config['PARAMS_DIR'])
    if not os.path.isdir(config['PLOT_DIR']):
        os.makedirs(config['PLOT_DIR'])

    features, t0, target_vals = daily_seismic_and_interpolated_pressure(seismic_df, pressure_df)
    features['cum_counts'] = target_vals
    print('Aggregated pressure and seismic data.')
    # Pass historical target values as input.
    if 'cum_counts' not in config['feature_names']:
        config['feature_names'].append('cum_counts')

    train_dset, val_dset, test_dset, x_scaler, y_scaler = construct_dataset(features, target_vals, config['seq_length'],
                                                                  config['feature_names'], config['train_frac'], config['val_frac'],
                                                                  do_normalize=True)
    print('Train-val-test split completed.')

    # Set up data loaders.
    train_loader = DataLoader(train_dset, config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dset, config['batch_size'])

    # Intialize model.
    model = ShallowRegLSTM(len(config['feature_names']), config['hidden_size'], config['num_layers'],
                           config['dropout'], config['monotonic_fn'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    print('LSTM model created and optimizer initialized.')

    # Model training
    start_time = time.time()
    n_epoch = np.arange(config['max_epoch'])
    train_loss = np.zeros(config['max_epoch'])
    val_loss = np.zeros(config['max_epoch'])
    val_dms_loss = np.zeros(config['max_epoch'])
    print('Beginning training')
    for idx_epoch in range(config['max_epoch']):
        print('EPOCH %d -------'% (idx_epoch))
        train_loss[idx_epoch] = train_model(train_loader, model, config['loss_function'], optimizer)
        val_loss[idx_epoch] = test_model(val_loader, model, config['loss_function'])
        # Unroll forecast on validation data.
        valtrain_forecast = unroll_forecast(model, train_dset, val_dset, config['seq_length'])
        # Loss between direct forecast and unbatched validation data
        val_dms_loss[idx_epoch] = config['loss_function'](valtrain_forecast.squeeze(), val_dset.Y)
        # Save model parameters to disk for every epoch of traning.
        torch.save(model.state_dict(), config['PARAMS_DIR']+'epoch%d.h5'% (idx_epoch))
        print()
    end_time = time.time()
    train_time = (end_time - start_time)/60.0 # Training time (minutes)
    print('Model training took %.2f minutes.'% (train_time))

    # Find epoch of robust fit.
    idx_robustfit = np.argmin(val_dms_loss)
    print('Loss on unbatched validation data was minimized at epoch %d.'% (idx_robustfit))
    # Load parameter values from epoch of robust fit.
    model.load_state_dict(torch.load(config['PARAMS_DIR']+'epoch%d.h5'% (idx_robustfit)))

    # Obtain unrolled model forecast on validation and test data.
    valtest_dset = concat_datasets(val_dset.X, val_dset.Y, test_dset.X, test_dset.Y, config['seq_length'], config['horizon_length'], config['feature_names'])
    valtest_forecast = unroll_forecast(model, train_dset, valtest_dset, config['seq_length'])
    val_forecast = valtest_forecast[:len(val_dset.Y)]
    test_forecast = valtest_forecast[len(val_dset.Y):]
    # Undo normalization.
    y_train = y_scaler.inverse_transform(train_dset.Y.numpy().reshape(-1,1)).squeeze()
    y_val = y_scaler.inverse_transform(val_dset.Y.numpy().reshape(-1,1)).squeeze()
    y_test = y_scaler.inverse_transform(test_dset.Y.numpy().reshape(-1,1)).squeeze()
    val_forecast = y_scaler.inverse_transform(val_forecast.numpy()).squeeze()
    test_forecast = y_scaler.inverse_transform(test_forecast.numpy()).squeeze()
    print('Computed model prediction on validation and test data')

    print('Plotting loss curve')
    plot_losscurve(n_epoch, train_loss, val_dms_loss, config['criterion'], config['PLOT_DIR']+'losscurve', config['plot_formats'])

    print('Plotting train/test data and model prediction')
    val_start_idx = len(train_dset.Y)
    test_start_idx = val_start_idx + len(val_dset.Y)
    plot_modelpred(features.days[:val_start_idx], y_train, features.days[val_start_idx:test_start_idx], y_val,
                   features.days[test_start_idx:], y_test, val_forecast, test_forecast,
                   t0, config['PLOT_DIR']+'prediction', config['plot_formats'])

    # Save loss curve data to file.
    losscurve_df = pd.DataFrame({'epoch_number':n_epoch, 'train_loss':train_loss, 'val_loss':val_loss, 'val_dms_loss': val_dms_loss})
    losscurve_df.to_csv(config['PARAMS_DIR']+'losscurve.csv',index=None)
    print('Loss curve data written to file.')

    # Save training data to file.
    out_train_df = pd.DataFrame({'days': features.days[:val_start_idx].values, 'y_data': y_train})
    out_train_df.to_csv(config['PARAMS_DIR']+'train.csv', index=None)
    print('Training data written to file.')

    # Save validation data and model predictions on validation data to file.
    out_val_df = pd.DataFrame({'days': features.days[val_start_idx:test_start_idx].values, 'y_data': y_val, 'y_forecast': val_forecast})
    out_val_df.to_csv(config['PARAMS_DIR']+'val.csv', index=None)
    print('Validation data and model forecast on validation data written to file.')

    # Save model predictions on test data to file.
    out_test_df = pd.DataFrame({'days': features.days[test_start_idx:].values, 'y_data': y_test, 'y_forecast': test_forecast})
    out_test_df.to_csv(config['PARAMS_DIR']+'/test.csv', index=None)
    print('Test data and model forecast on test data written to file.')

####################################################################
if __name__ == "__main__":
    fit_lstm(config)
####################################################################
