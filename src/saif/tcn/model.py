from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from darts.models import TCNModel
from darts.models.forecasting.tcn_model import _TCNModule

# Ordering taken from here:
# https://github.com/unit8co/darts/blob/master/darts/models/forecasting/pl_forecasting_module.py#L416
class _PositiveTCNModule(_TCNModule):
    def forward(self, x_in):
        x, _ = x_in
        out = super().forward(x_in)

        out = torch.sqrt(abs(out))

        out = out.cumsum(1)
        out = out / out.shape[1] ** 2

        out = out + x[:, self.target_length - 1, 0, None, None, None]

        return out
    
class PositiveTCNModel(TCNModel):
    def _create_model(self, train_sample: Tuple[torch.Tensor]) -> torch.nn.Module:
        # samples are made of (past_target, past_covariates, future_target)
        input_dim = train_sample[0].shape[1] + (
            train_sample[1].shape[1] if train_sample[1] is not None else 0
        )
        output_dim = train_sample[-1].shape[1]
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        return _PositiveTCNModule(
            input_size=input_dim,
            target_size=output_dim,
            nr_params=nr_params,
            kernel_size=self.kernel_size,
            num_filters=self.num_filters,
            num_layers=self.num_layers,
            dilation_base=self.dilation_base,
            target_length=self.output_chunk_length,
            dropout=self.dropout,
            weight_norm=self.weight_norm,
            **self.pl_module_params,
        )