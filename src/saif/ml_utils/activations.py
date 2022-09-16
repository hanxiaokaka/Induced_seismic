import torch
import torch.nn.functional as F

MONOTONIC_FUNCS = {
    'abs' : lambda x : x.abs(),
    'quad' : lambda x : x ** 2,
    'relu' : F.relu,
    'exp' : torch.exp,
    'sigmoid' : torch.sigmoid,
    'no_func' : lambda x : x,
    'sqrt' : lambda x : torch.sqrt(abs(x))
}
