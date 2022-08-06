from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def plot_losscurve(n_epoch, train_loss, val_loss, criterion='MSE', basename='losscurve', output_formats=['.png']):
    '''
    Plot loss curve for training and validation data.

    Parameters:
    --------------
    n_epochs: (1D NumPy array) Epoch numbers
    train_loss: (1D NumPy array) Loss values on training data
    val_loss: (1D NumPy array) Loss values on validation data
    criterion: (string) Label for adopted loss critertion
    basename: (string) Plot basename including path
    output_formats: (list of strings) Plot output formats
    '''
    fig = plt.figure()
    plt.semilogy(n_epoch, train_loss, '-k', label='Training data')
    plt.semilogy(n_epoch, val_loss, '-b', label='Validation data')
    plt.xlabel('Epoch number', fontsize=14)
    plt.ylabel('%s loss'% (criterion), fontsize=14)
    plt.legend(loc='best', prop={'size':14})
    plt.grid(linestyle=':', alpha=0.7)
    plt.xlim((0,np.max(n_epoch)))
    plt.tight_layout()
    for format in output_formats:
        plt.savefig(basename+format)
    plt.close()

def plot_modelpred(x_train, y_train, x_val, y_val, x_test, y_test, pred_val, pred_test, t0, basename, output_formats):
    '''
    Show train-test split and model predictions on training/test data.

    Parameters:
    --------------
    x_train: (1D NumPy array) Values of the independent variable covered in training data
    y_train: (1D NumPy array) Values of the target variable covered in training data
    x_val: (1D NumPy array) Values of the independent variable covered in validation data
    y_val: (1D NumPy array) Values of the target variable covered in validation data
    x_test: (1D NumPy array) Values of the independent variable covered in test data
    y_test: (1D NumPy array) Values of the target variable covered in test data
    pred_val: (1D NumPy array) Model prediction on validation data
    pred_test: (1D NumPy array) Model prediction on test data
    t0: (float) Start epoch of training data in seconds
    basename: (string) Plot basename including path
    output_formats: (list of strings) Plot output formats
    '''
    min_date = datetime.fromtimestamp(t0)
    fig = plt.figure()
    line1, = plt.plot(x_train, y_train, '-k')
    line2, = plt.plot(x_val, y_val, '-b')
    line3, = plt.plot(x_val, pred_val, ':b')
    line4, = plt.plot(x_test, y_test, '-r')
    line5, = plt.plot(x_test, pred_test, ':r')
    plt.xlabel('Time (days) since %s'% (min_date.strftime('%Y - %m - %d')), fontsize=14)
    plt.ylabel('Cumulative earthquake count', fontsize=14)
    first_legend = plt.legend(handles=[line1, line2, line4], labels=['Training data', 'Validation data', 'Test data'], loc='upper left', prop={'size':12})
    plt.gca().add_artist(first_legend)
    second_legend = plt.legend(handles=[line2, line3], labels=['Data', 'Forecast'], loc='lower right', prop={'size':12})
    plt.grid(linestyle=':', alpha=0.7)
    plt.tight_layout()
    for format in output_formats:
        plt.savefig(basename+format)
    plt.close()
