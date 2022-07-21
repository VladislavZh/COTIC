import torch
import torch.nn as nn
import copy

from .cont_cnn_dense_layers import ContConv1dDense, ContConv1dDenseSim
from .kernels import Kernel


class CCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        nb_filters: int,
        nb_layers: int,
        num_types: int,
        hidden_1: int,
        hidden_2: int,
        hidden_3: int
    ) -> None:
        self.event_emb = nn.Embedding(num_types + 1, in_channels, padding_idx=0)
        
        self.in_channels = [in_channels] + [nb_filters] * nb_layers
        skip_connections = [False] + [True] * nb_layers
        include_zero_lag = [False] + [True] * nb_layers
        
        self.convs = nn.ModuleList([ContConv1dDense(Kernel(hidden_1, hidden_2, hidden_3, self.in_channels[i], nb_filters), kernel_size, self.in_channels[i], nb_filters, include_zero_lag[i], skip_connections[i]) for i in range(nb_layers)])
        
        self.final = nn.Sequential(ContConv1dDenseSim(Kernel(hidden_1, hidden_2, hidden_3, nb_filters, nb_filters), kernel_size, nb_filters, nb_filters), nn.ReLU(), nn.Linear(nb_filters, num_types), nn.Softplus())
        
    def forward(
        self,
        event_times: torch.Tensor,
        event_types: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass that computes self.convs and return encoder output
        
        args:
            event_times - torch.Tensor, shape = (bs, L) event times
            event_types - torch.Tensor, shape = (bs, L) event types
            lengths - torch.Tensor, shape = (bs,) sequence lengths
        """
        
        enc_output = self.event_emb(event_types)
        
        for conv in self.convs:
            enc_output = torch.nn.functional.leaky_relu(conv(event_times, enc_output, lengths),0.1)
            
        return enc_output
        