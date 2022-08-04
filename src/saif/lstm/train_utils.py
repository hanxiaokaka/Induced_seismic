import torch
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
