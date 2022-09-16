import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from darts.logging import get_logger, raise_if_not
from darts.models.forecasting.pl_forecasting_module import PLPastCovariatesModule
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel
from darts.timeseries import TimeSeries
from darts.utils.data import PastCovariatesShiftedDataset
from darts.utils.torch import MonteCarloDropout

import saif.scinet.model as scinet

logger = get_logger(__name__)

from saif.ml_utils.activations import MONOTONIC_FUNCS as mfuncs


class _SCINetModule(PLPastCovariatesModule):
    def __init__(
        self,
        input_size: int,
        kernel_size: int,
        target_size: int,
        target_length: int,
        num_levels: int,
        hidden_size: int,
        num_blocks: int,
        groups: int,
        dropout: float,
        weight_norm: bool,
        **kwargs
    ):
        super().__init__(**kwargs)


        self.input_size = input_size
        self.kernel_size = kernel_size
        
        self.target_size = target_size
        self.target_length = target_length

        self.num_levels = num_levels
        self.hidden_size = hidden_size

        self.groups = groups
        self.dropout = dropout

        self.num_blocks = num_blocks
        self.weight_norm = weight_norm
        
        block_list = []
        for _ in range(num_blocks):
            _block = scinet.EncoderTree(
                in_planes=self.input_size,
                num_levels=self.num_levels,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
                groups=self.groups,
                hidden_size=self.hidden_size,
                INN=True
            )
            block_list.append(_block)
        self.block_list = nn.ModuleList(block_list)

        self.channel_projector = nn.Conv1d(
            self.input_size, self.target_size, kernel_size=1, stride=1, bias=True
        )

        if weight_norm:
            self.channel_projector = nn.utils.weight_norm(
                self.channel_projector
            )

    def forward(self, x_in: Tuple):
        x, _ = x_in
        # data is of size (batch_size, input_chunk_length, input_size)
        #batch_size = x.size(0)
        #out = x.permute(0, 2, 1)
        out = x
        #out = self.bn1(out)
        #out = out.permute(0, 2, 1)

        #print('in', out.shape)
        for _block in self.block_list:
            block_out = _block(out)
            out = out + block_out
            out = F.relu(out)

        #print('scinet block', out.shape)
        out = out.permute(0, 2, 1)
        out = self.channel_projector(out)
        out = out.permute(0, 2, 1)

        #print('projector', out.shape)
        out = torch.sqrt(abs(out))

        out = out.cumsum(1)
        out = out / out.shape[1] ** 2
        #print(out.shape)
        out = out + x[:, self.target_length - 1, 0, None, None]

        #print('final with addition', out.shape)

        out = out.unsqueeze(dim=-1)
        
        return out

    @property
    def first_prediction_index(self) -> int:
        return -self.output_chunk_length


class SCINetModel(PastCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        num_levels: int = 2,
        hidden_size: int = 2,
        groups: int = 1,
        kernel_size: int = 3,
        dropout: float = 0.2,
        num_blocks:int = 1,
        weight_norm: bool = False,
        **kwargs
    ):

        """
        SCINet with TCN
        """

        raise_if_not(
            kernel_size < input_chunk_length,
            "The kernel size must be strictly smaller than the input length.",
            logger,
        )
        raise_if_not(
            output_chunk_length < input_chunk_length,
            "The output length must be strictly smaller than the input length",
            logger,
        )
        raise_if_not(
            num_blocks >= 1,
            "The number of blocks must be greater than or equal to 1",
            logger,
        )


        super().__init__(**self._extract_torch_model_params(**self.model_params))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)
        
        self.num_levels = num_levels
        self.hidden_size = hidden_size
        self.groups = groups
        self.kernel_size = kernel_size
        self.dropout = dropout

        self.num_blocks = num_blocks
        self.weight_norm = weight_norm

    def _create_model(self, train_sample: Tuple[torch.Tensor]) -> torch.nn.Module:
        # samples are made of (past_target, past_covariates, future_target)
        input_dim = train_sample[0].shape[1] + (
            train_sample[1].shape[1] if train_sample[1] is not None else 0
        )
        output_dim = train_sample[-1].shape[1]
        #TODO: likelihood
        
        return _SCINetModule(
            input_size=input_dim,
            target_size=output_dim,
            kernel_size=self.kernel_size,
            num_levels=self.num_levels,
            hidden_size=self.hidden_size,
            groups=self.groups,
            target_length=self.output_chunk_length,
            dropout=self.dropout,
            num_blocks=self.num_blocks,
            weight_norm=self.weight_norm,
            **self.pl_module_params
        )

    def _build_train_dataset(
        self,
        target: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        max_samples_per_ts: Optional[int],
    ) -> PastCovariatesShiftedDataset:

        return PastCovariatesShiftedDataset(
            target_series=target,
            covariates=past_covariates,
            length=self.input_chunk_length,
            shift=self.output_chunk_length,
            max_samples_per_ts=max_samples_per_ts#,
            #The reference code has this, but it throws an error
            #Commented for now
            #use_static_covariates=self._supports_static_covariates(),
        )

    @staticmethod
    def _supports_static_covariates() -> bool:
        return False