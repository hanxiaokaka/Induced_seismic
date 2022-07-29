import os, time
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from saif.scinet.dataset import normalize
from saif.lstm.data_utils import bin_data, train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from dataset import daily_seismic_and_interpolated_pressure, overlap_and_interpolate, aggregate_seismic
####################################################################
# INPUTS

location = 'Decatur' # Location tag
seismic_df = pd.read_csv('gs://us-geomechanicsforco2-dev-staging/temporal_datasets/decatur_illinois/seismic.csv')
# Retain only portion of seismic data that overlaps with the pressure data.
pressure_df = pd.read_csv('gs://us-geomechanicsforco2-dev-staging/temporal_datasets/decatur_illinois/pressure.csv')
seismic_df = seismic_df[seismic_df['epoch'] < pressure_df['epoch'].max()]
# Define target and feature columns.
target = 'cum_counts'
features = ['epoch']
# No. of time samples desired across binned data.
N_samples = 1110
# Specify horizon length.
horizon_length = 222
# Hyperparameters for xgboost
params = {'objective': 'reg:pseudohubererror', 'n_estimators': 40000, 'max_depth': 8,
          'learning_rate': 1e-4, 'subsample': 0.7}
# Path to save model parameters at each epoch of training
PARAMS_DIR = '../../data/06_models/xgboost/'
# Path to plots
PLOT_DIR = '../../plots/xgboost/'
# Output plot formats
plot_formats = ['.png']
####################################################################
if not os.path.isdir(PARAMS_DIR):
    os.makedirs(PARAMS_DIR)
if not os.path.isdir(PLOT_DIR):
    os.makedirs(PLOT_DIR)
####################################################################
# PROCESSING

# Bin earthquake data.
counts_df = bin_data(seismic_df, N_samples=N_samples)
print('Earthquake data binned to time series.')

# Earliest date in seismic data
date0 = datetime.fromtimestamp(counts_df['epoch'].min())

train_df, test_df = train_test_split(counts_df, horizon_len=horizon_length)
print('Length of training data = %d samples'% (len(train_df)))
print('Length of test data =  %d samples'% (len(test_df)))

X_scaler = RobustScaler()
y_scaler = RobustScaler()
# Fit and transform training data.
X_train_norm = X_scaler.fit_transform(train_df[features].to_numpy())
y_train_norm = y_scaler.fit_transform(train_df[target].to_numpy().reshape(-1,1)).squeeze()
# Use statistics of the training data to normalize test data.
X_test_norm = X_scaler.transform(test_df[features].to_numpy())
y_test_norm = y_scaler.transform(test_df[target].to_numpy().reshape(-1,1)).squeeze()
print('Data normalization completed.')

# Initialize model.
xgbr = XGBRegressor(objective=params['objective'], n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'], eta=params['learning_rate'],
                    subsample=params['subsample'])
# Evaluation set for loss curve computation
eval_set = [(X_train_norm, y_train_norm), (X_test_norm, y_test_norm)]
# Fit model on training data.
xgbr.fit(X_train_norm, y_train_norm, eval_set=eval_set, verbose=False)
print('xgboost model fit completed.')

# Inverse transformations on training data
X_train = X_scaler.inverse_transform(X_train_norm).squeeze()
X_train_yr = (X_train - counts_df['epoch'].min())/(60*60*24*365.25) # s -> year
y_train = y_scaler.inverse_transform(y_train_norm.reshape(-1,1)).squeeze()
# Inverse transformations on test data
X_test = X_scaler.inverse_transform(X_test_norm).squeeze()
X_test_yr = (X_test - counts_df['epoch'].min())/(60*60*24*365.25) # s -> year
y_test = y_scaler.inverse_transform(y_test_norm.reshape(-1,1)).squeeze()

# Obtain model predictions on training and test data.
pred_train = y_scaler.inverse_transform(xgbr.predict(X_train_norm).reshape(-1,1)).squeeze()
pred_test = y_scaler.inverse_transform(xgbr.predict(X_test_norm).reshape(-1,1)).squeeze()

# Loss curve data
results = xgbr.evals_result()
boost_iter = np.arange(params['n_estimators']) + 1
train_loss = results["validation_0"]['mphe']
test_loss = results["validation_1"]['mphe']

# Plot model prediction.
fig = plt.figure()
plt.plot(X_train_yr, y_train, '-k')
plt.plot(X_test_yr, y_test, '-r')
plt.plot(X_train_yr, pred_train, ':k')
plt.plot(X_test_yr, pred_test, ':r')
plt.xlabel('Time (years) since %s'% (date0.strftime('%Y - %m - %d')), fontsize=14)
plt.ylabel('Cumulative earthquake count', fontsize=14)
plt.grid(linestyle=':', alpha=0.7)
plt.tight_layout()
for format in plot_formats:
    plt.savefig(PLOT_DIR + '/pred_est%d_maxdepth%d'% (params['n_estimators'], params['max_depth']) + format)
plt.close()

# Plot training deviance.
fig = plt.figure()
plt.semilogy(boost_iter, train_loss, '-k',label="Training data")
plt.semilogy(boost_iter, test_loss, '-r', label="Test data")
plt.legend(loc='best', prop={'size':14})
plt.xlabel('No. of boosting iterations', fontsize=14)
plt.ylabel('Pseudohuber loss', fontsize=14)
plt.grid(linestyle=':', alpha=0.7)
plt.tight_layout()
for format in plot_formats:
    plt.savefig(PLOT_DIR + '/loss_est%d_maxdepth%d'% (params['n_estimators'], params['max_depth']) + format)
plt.close()
####################################################################
