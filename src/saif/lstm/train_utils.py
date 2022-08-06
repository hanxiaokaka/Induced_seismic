import torch
from torch import nn
from saif.scinet.dataset import TimeSeriesDataset
####################################################################
def train_model(data_loader, model, loss_function, optimizer):
    '''
    Conduct model training.

    Parameters:
    -------------
    data_loader: torch.utils.data.DataLoader object
    model: nn.Module (or PyTorch model)
    loss_function: Loss function
    optimizer: Optimizer selected from torch.optim
    '''
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    # Loop through various batches in data loader.
    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)
        optimizer.zero_grad() # Set gradients to zero.
        loss.backward() # Compute gradients
        optimizer.step() # Update model parameters
        total_loss += loss.item()
    # Compute batch-averaged loss.
    avg_loss = total_loss / num_batches
    print('Training loss: %s'% (avg_loss))
    return avg_loss

####################################################################
def test_model(data_loader, model, loss_function):
    '''
    Evaluate model on test data and compute associated loss.

    Parameters:
    -------------
    data_loader: torch.utils.data.DataLoader object
    model: nn.Module (or PyTorch model)
    loss_function: Loss function
    '''
    num_batches = len(data_loader)
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()
    avg_loss = total_loss / num_batches
    print('Test loss: %s'% (avg_loss))
    return avg_loss

####################################################################
def predict(data_loader, model):
    '''
    Obtain model prediction on data.

    Parameters:
    -------------
    data_loader: torch.utils.data.DataLoader object
    model: nn.Module (or PyTorch model)
    '''
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)
    return output
####################################################################
def unroll_forecast(model: nn.Module, train_dset: TimeSeriesDataset, test_dset: TimeSeriesDataset, seq_length: int) -> torch.tensor:
    '''
    Unroll model forecast over x-ranges of test data. Presumes a horizon length of 1 sample.
    Also, assumes that the last column in train_dset.X and test_dset.X stores historical values of the target variable.

    Parameters:
    -------------
    model: nn.Module (or PyTorch model)
        PyTorch machine learning model

    train_dset: TimeSeriesDataset object
        Training data

    test_dset: TimeSeriesDataset object
        Test data

    seq_length: int
        Sequence length

    Returns:
    -------------
    forecast_y: torch.tensor
        Unrolled model forecast over x-ranges of test data
    '''
    model.eval()
    forecast_X = torch.cat((torch.clone(train_dset.X[-seq_length:]),torch.clone(test_dset.X)))
    sample_x = torch.clone(train_dset.X[-seq_length:])
    # Shape of sample_x = (Sequence length, N_features)
    # sample_x[:, -1] is the historical output time series.
    forecast_y = torch.tensor([])
    with torch.no_grad():
        # Loop over number of forecast horizons.
        for i in range(len(test_dset.Y)):
            y_star = model(sample_x[None,:,:])
            forecast_y = torch.cat((forecast_y, y_star))
            forecast_X[seq_length+i,-1] = y_star
            # Move sample_x window forward alone forecast_X by horizon length.
            sample_x = torch.clone(forecast_X[i+1:i+1+seq_length])
    return forecast_y
####################################################################
