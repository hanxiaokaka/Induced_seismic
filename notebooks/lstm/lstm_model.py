import os, time
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import pandas as pd
import torch
from torch import  nn
from torch.utils.data import DataLoader
from saif.lstm.data_utils import bin_data, train_test_split, standardize
from saif.lstm.dataset import SeismicDataset
from saif.lstm.model import ShallowRegLSTM
from saif.lstm.train_utils import train_model, test_model, predict
from saif.lstm.plot_utils import plot_losscurve, plot_modelpred
####################################################################
# INPUTS

location = 'Decatur' # Location tag
seismic_df = pd.read_csv('gs://us-geomechanicsforco2-dev-staging/temporal_datasets/decatur_illinois/seismic.csv')
pressure_df = pd.read_csv('gs://us-geomechanicsforco2-dev-staging/temporal_datasets/decatur_illinois/pressure.csv')
seismic_df = seismic_df[seismic_df['epoch'] < pressure_df['epoch'].max()]
# Define target and feature columns.
target = 'cum_counts'
features = ['epoch']
# No. of time samples desired across binned data.
N_samples = 1110
# Specify sequence and horizon lengths.
horizon_len = 222
seq_length = 16
# Set batch size for training.
batch_size = 32
# Number of hidden units in LSTM (preferably a power of 2)
N_hidden = 16
# Select loss function.
criterion = 'Huber'
loss_function = nn.HuberLoss()
# Specify learning rate.
learning_rate = 1.0e-5
# Max no. of epochs of traning
max_epoch = 4000
# Path to save model parameters at each epoch of training
PARAMS_DIR = '../../data/06_models/lstm/horizon_' + str(horizon_len) +'/'
# Path to plots
PLOT_DIR = '../../plots/lstm/'
# Output plot formats
plot_formats = ['.png']
####################################################################
if not os.path.isdir(PARAMS_DIR):
    os.makedirs(PARAMS_DIR)
if not os.path.isdir(PLOT_DIR):
    os.makedirs(PLOT_DIR)
####################################################################
# PROCESSING

counts_df = bin_data(seismic_df, N_samples=N_samples)
print('Earthquake data binned to time series.')

train_df, test_df = train_test_split(counts_df, horizon_len=horizon_len)
print('Length of training data = %d samples'% (len(train_df)))
print('Length of test data =  %d samples'% (len(test_df)))

train_df_normed, test_df_normed, target_mean, target_stdev = standardize(train_df, test_df, target=target)
print('Normalization done.')

# Create data sets.
train_dataset = SeismicDataset(train_df_normed, features, target, seq_length)
test_dataset = SeismicDataset(test_df_normed, features, target, seq_length)

# Set up data loaders.
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader =  DataLoader(test_dataset, batch_size)
X, y = next(iter(train_loader))
print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Initialize LSTM model.
model = ShallowRegLSTM(len(features), N_hidden)
# Select optimizer and initial learning rate.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print('LSTM model created and optimizer initialized.')

# Model training
start_time = time.time()
n_epoch = np.arange(max_epoch)+1
train_loss = np.zeros(max_epoch)
test_loss = np.zeros(max_epoch)
print('Beginning training')
for idx_epoch in range(max_epoch):
    print('EPOCH %d -------'% (idx_epoch+1))
    train_loss[idx_epoch] = train_model(train_loader, model, loss_function, optimizer)
    test_loss[idx_epoch] = test_model(test_loader, model, loss_function)
    # Save model parameters to disk for every epoch of traning.
    torch.save(model.state_dict(), PARAMS_DIR+'epoch%d.h5'% (idx_epoch+1))
    print()
end_time = time.time()
train_time = (end_time - start_time)/60.0 # Training time (minutes)
print('Model training took %.2f minutes.'% (train_time))

# Find epoch of robust fit.
idx_robustfit = np.argmin(test_loss)
print('Loss on test data was minimized at epoch %d.'% (idx_robustfit+1))
# Load parameter values from epoch of robust fit.
model.load_state_dict(torch.load(PARAMS_DIR+'epoch%d.h5'% (idx_robustfit+1)))

# Obtain model prediction on test data.
train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
prediction_train = predict(train_eval_loader, model)
prediction_test = predict(test_loader, model)
# Undo normalization.
prediction_train = (prediction_train*target_stdev) + target_mean
prediction_test =  (prediction_test*target_stdev) + target_mean
print('Computed model prediction on training and test data')

print('Plotting loss curve')
plot_losscurve(n_epoch, train_loss, test_loss, criterion, PLOT_DIR+location+'_loss_horizon' + str(horizon_len), plot_formats)

print('Plotting train/test data and model prediction')
plot_modelpred((train_df['epoch']-train_df['epoch'].min())/(60*60*24*365.25),
               train_df['cum_counts'],
               (test_df['epoch']-train_df['epoch'].min())/(60*60*24*365.25),
               test_df['cum_counts'], prediction_train, prediction_test,
               PLOT_DIR+location+'_pred_horizon' + str(horizon_len), plot_formats)

# Save loss curve data to file.
losscurve_df = pd.DataFrame({'epoch':n_epoch, 'train_loss':train_loss, 'test_loss':test_loss})
losscurve_df.to_csv(PARAMS_DIR+'/lstm_loss_horizon%d_seq%d.csv'% (horizon_len, seq_length),index=None)
print('Loss curve data written to file.')

# Save model predictions for best fit to file.
pred_df = pd.DataFrame({'x_test':test_df['epoch'], 'y_test':test_df['cum_counts'], 'pred_test':prediction_test
                       })
pred_df.to_csv(PARAMS_DIR+'/lstm_pred_horizon%d_seq%d.csv'% (horizon_len, seq_length), index=None)
print('Model predictions on test data written to file.')
####################################################################
