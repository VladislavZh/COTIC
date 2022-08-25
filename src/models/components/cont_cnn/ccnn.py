import torch
import torch.nn as nn
import copy

from .cont_cnn_layers import ContConv1d, ContConv1dSim
from .kernels import Kernel,LinearKernel


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
        super().__init__()
        self.event_emb = nn.Embedding(num_types + 2, in_channels, padding_idx=0)
        
        self.in_channels = [in_channels] + [nb_filters] * nb_layers
        skip_connections = [False] + [True] * nb_layers
        include_zero_lag = [True] + [True] * nb_layers
        self.dilation_factors = [2 ** i for i in range(0, nb_layers)]

        self.num_types = num_types
        
        self.convs = nn.ModuleList([ContConv1d(Kernel(hidden_1, hidden_2, hidden_3, self.in_channels[i], nb_filters), kernel_size, self.in_channels[i], nb_filters, self.dilation_factors[i], include_zero_lag[i], skip_connections[i]) for i in range(nb_layers)])
        
        self.final_list = nn.ModuleList([ContConv1dSim(Kernel(hidden_1, hidden_2, hidden_3, nb_filters, nb_filters), 1, nb_filters, nb_filters), nn.ReLU(), nn.Linear(nb_filters, num_types), nn.Softplus()])
        
    def __add_bos(self, event_times, event_types, lengths):
        bs, L = event_times.shape
        event_times = torch.concat([torch.zeros(bs, 1).to(event_times.device), event_times], dim = 1)
        max_event_type = torch.max(event_types) + 1
        tmp = (torch.ones(bs,1).to(event_types.device) * max_event_type).long()
        event_types = torch.concat([tmp, event_types], dim = 1)
        lengths += 1
        return event_times, event_types, lengths
        
    def forward(
        self,
        event_times: torch.Tensor,
        event_types: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass that computes self.convs and return encoder output
        
        args:
            event_times - torch.Tensor, shape = (bs, L) event times
            event_types - torch.Tensor, shape = (bs, L) event types
            lengths - torch.Tensor, shape = (bs,) sequence lengths
        """
        lengths = torch.sum(event_types.ne(0).type(torch.float), dim = 1).long()
        event_times, event_types, lengths = self.__add_bos(event_times, event_types, lengths)
        
        non_pad_mask = event_types.ne(0)
        
        enc_output = self.event_emb(event_types)

        for conv in self.convs:
            enc_output = torch.nn.functional.leaky_relu(conv(event_times, enc_output, non_pad_mask),0.1)
            
        return enc_output
        
    def final(self, times, true_times, true_features, non_pad_mask, sim_size):
        out = self.final_list[0](times, true_times, true_features, non_pad_mask, sim_size)
        for layer in self.final_list[1:]:
            out = layer(out)
        return out/100
