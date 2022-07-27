import numpy as np
import matplotlib.pyplot as plt

def plot_losscurve(n_epoch, train_loss, test_loss, criterion='MSE', basename='losscurve', output_formats=['.png']):
    '''
    Plot loss curve for training and test data.

    Parameters:
    --------------
    n_epochs: (1D NumPy array) Epoch numbers
    train_loss: (1D NumPy array) Loss values on training data
    test_loss: (1D NumPy array) Loss values on test data
    criterion: (string) Label for adopted loss critertion
    basename: (string) Plot basename including path
    output_formats: (list of strings) Plot output formats
    '''
    fig = plt.figure()
    plt.semilogy(n_epoch, train_loss, '-k', label='Training data')
    plt.semilogy(n_epoch, test_loss, '-r', label='Test data')
    plt.xlabel('Epoch number', fontsize=14)
    plt.ylabel('Batch-averaged %s loss'% (criterion), fontsize=14)
    plt.legend(loc='best', prop={'size':14})
    plt.grid(linestyle=':', alpha=0.7)
    plt.xlim((0,np.max(n_epoch)))
    plt.tight_layout()
    for format in output_formats:
        plt.savefig(basename+format)
    plt.close()

def plot_modelpred(x_train, y_train, x_test, y_test, pred_train, pred_test, basename, output_formats):
    '''
    Show train-test split and model predictions on training/test data.

    Parameters:
    --------------
    x_train: (1D NumPy array) Values of the independent variable covered in training data
    y_train: (1D NumPy array) Values of the target variable covered in training data
    x_test: (1D NumPy array) Values of the independent variable covered in test data
    y_test: (1D NumPy array) Values of the target variable covered in test data
    pred_train: (1D NumPy array) Model prediction on training data
    pred_test: (1D NumPy array) Model prediction on test data
    basename: (string) Plot basename including path
    output_formats: (list of strings) Plot output formats
    '''
    fig = plt.figure()
    plt.plot(x_train, y_train, '-k')
    plt.plot(x_test, y_test, '-r')
    plt.plot(x_train, pred_train, ':k')
    plt.plot(x_test, pred_test, ':r')
    plt.xlabel('Time (years)', fontsize=14)
    plt.ylabel('Cumulative earthquake count', fontsize=14)
    plt.grid(linestyle=':', alpha=0.7)
    plt.tight_layout()
    for format in output_formats:
        plt.savefig(basename+format)
    plt.close()
