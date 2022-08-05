import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def collective_loss(mask, masked_output, target, scale=None):
    n_site, n_batch, n_steps = masked_output.shape
    
    mask = (target != -1).float()
    N = mask.sum(-1).sum(-1) # number of samples per site
    
    masked_target = mask * target
    
    if scale is not None:
        masked_output = masked_output / scale
        masked_target = masked_target / scale
        
    diffs = F.huber_loss(
        masked_output, masked_target, reduction='none'
    ) # [n_site, n_batch, n_steps]
    
    mean_diffs = diffs.sum(-1).sum(-1) / N
    
    return mean_diffs.sum()

def generate_init_pre_params(init_params):
    return torch.logit(init_params)

def generate_train_loss_fn(
        signal_sections, signal_starts,
        model, lower_bounds, upper_bounds, scale
    ):
    mask = (signal_sections['number'] != -1).float()
    proj = ParamProjector(lower_bounds, upper_bounds)

    #TODO:
    #Add GPU device
    
    def loss_fn(pre_params):
        params = proj(pre_params)
        _, Nt = model.masked_forward(
            params, 
            p=signal_sections['pressure'][:, :-1, :], 
            dpdt=signal_sections['dpdt'][:, :-1, :], 
            delta_t=signal_sections['delta_t'][:, :-1, :], 
            R0=signal_starts['rate'][:, :-1],
            mask=mask[:, :-1, :]
        )

        loss = collective_loss(
            mask=mask[:, :-1, :],
            masked_output=Nt, 
            target=signal_sections['number'][:, :-1, :], 
            scale=scale
        )

        return loss
    
    return proj, loss_fn

def generate_test_loss_fn(
        signal_sections, signal_starts, model, proj, scale
    ):
    mask = (signal_sections['number'] != -1).float()
    number = signal_sections['number'][:, -1, None, :]
    
    def loss_fn(pre_params):
        with torch.no_grad():
            params = proj(pre_params)
            _, Nt = model.masked_forward(
                params, 
                p=signal_sections['pressure'], 
                dpdt=signal_sections['dpdt'], 
                delta_t=signal_sections['delta_t'], 
                R0=signal_starts['rate'],
                mask=mask
            )
            Nt = Nt[:, -1, None, :]
            _loss = F.huber_loss(Nt / scale, number / scale)
        return _loss

    return loss_fn

#TODO:
#Add docstrings
#Add independent loss

class ParamProjector(nn.Module):
    def __init__(self, lower_bounds, upper_bounds):
        """
        lower_bounds, upper_bounds: [3]
        [mu_minus_alpha, rate_coeff, rate_factor]
        """
        super().__init__()
        
        self.register_buffer('lb', lower_bounds)
        self.register_buffer('ub', upper_bounds)
        
    def forward(self, pre_params):
        out = self.lb + (self.ub - self.lb) * torch.sigmoid(pre_params)
        
        return out

    def inverse(self, params):
        return torch.logit( (params - self.lb) / (self.ub - self.lb)  )


#TODO:
#Add DeltaCFS Params in here as well.

class ValidationCallback():
    def __init__(self, train_loss_fn, test_loss_fn):
        self.fn_train = train_loss_fn
        self.fn_test = test_loss_fn
        self.train_log = []
        self.test_log = []
        
    def __call__(self, x):
        self.train_log.append(self.fn_train(x).item())
        self.test_log.append(self.fn_test(x).item())