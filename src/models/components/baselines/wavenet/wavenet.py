import torch
import torch.nn as nn
import torch.nn.init as init
from .causal_conv_1d import DilatedCausalConv1d

import math

from typing import Tuple, Any


class WaveNet(nn.Module):
    """Core WaveNet model."""

    def __init__(self, hyperparams: dict, in_channels: int, num_types: int) -> None:
        """Initialize WaveNet model.

        args:
            hyperparams - dictionary with hyper-parameters (from config)
            in_channels - number of input channels
            num_types - number of event types in the datset
        """
        super().__init__()

        def weights_init(m: Any):
            """Initialize model's weights."""
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight.data)
                init.zeros_(m.bias.data)

        self.position_vec = torch.tensor(
            [
                math.pow(10000.0, 2.0 * (i // 2) / in_channels)
                for i in range(in_channels)
            ]
        )

        self.event_emb = nn.Embedding(num_types + 1, in_channels, padding_idx=0)

        self.dilation_factors = [2**i for i in range(0, hyperparams["nb_layers"])]
        self.in_channels = [in_channels] + [
            hyperparams["nb_filters"] for _ in range(hyperparams["nb_layers"])
        ]
        self.num_types = num_types
        self.dilated_causal_convs = nn.ModuleList(
            [
                DilatedCausalConv1d(
                    hyperparams, self.dilation_factors[i], self.in_channels[i]
                )
                for i in range(hyperparams["nb_layers"])
            ]
        )
        for dilated_causal_conv in self.dilated_causal_convs:
            dilated_causal_conv.apply(weights_init)

        self.output_layer_type = nn.Conv1d(
            in_channels=self.in_channels[-1], out_channels=self.num_types, kernel_size=1
        )
        self.output_layer_time = nn.Conv1d(
            in_channels=self.in_channels[-1], out_channels=1, kernel_size=1
        )
        self.output_layer_type.apply(weights_init)
        self.output_layer_time.apply(weights_init)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def temporal_enc(
        self, time: torch.Tensor, non_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """Temporal encoding of event sequences.

        args:
            time - true event times
            non_pad_mask - boolean mask indicating true event times

        returns:
            result - encoded times tensor
        """
        result = time.unsqueeze(-1) / self.position_vec.to(time.device)
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    @staticmethod
    def get_non_pad_mask(seq: torch.Tensor) -> torch.Tensor:
        """Get the non-padding positions (positions of true event)."""

        assert seq.dim() == 2
        return seq.ne(0).type(torch.float).unsqueeze(-1)

    def forward(
        self, event_time: torch.Tensor, event_type: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fowrard pass through the model.

        args:
            event_time - batch of sequences with event times
            event_type - batch of sequences with event types

        return:
            time_pred - predicted event times
            type_pred - predicted event types
        """
        non_pad_mask = self.get_non_pad_mask(event_type)

        temporal_enc = self.temporal_enc(event_time, non_pad_mask)
        event_emb = self.event_emb(event_type)

        x = temporal_enc + event_emb

        x = x.transpose(1, 2)

        for dilated_causal_conv in self.dilated_causal_convs:
            x = dilated_causal_conv(x)
        time_pred = (
            self.leaky_relu(self.output_layer_time(x)).transpose(1, 2) * non_pad_mask
        )
        type_pred = (
            self.leaky_relu(self.output_layer_type(x)).transpose(1, 2) * non_pad_mask
        )

        return time_pred, type_pred
