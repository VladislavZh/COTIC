import torch
import torch.nn as nn
import torch.nn.init as init
from .causal_conv_1d import DilatedCausalConv1d
from .interpolator import IntensPredictor

import math


class WaveNetIntens(nn.Module):
    def __init__(
        self,
        hyperparams: dict,
        in_channels: int,
        num_types: int,
        interpolator: nn.Module
    ) -> None:
        super().__init__()

        self.hidden_size = hyperparams['hidden_size']
        self.num_types = num_types

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight.data)
                init.zeros_(m.bias.data)

        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / in_channels) for i in range(in_channels)])
        
        self.position_vec_future = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / hyperparams['hidden_size']) for i in range(hyperparams['hidden_size'])])
        
        self.event_emb = nn.Embedding(num_types + 1, in_channels, padding_idx=0)
                
        self.dilation_factors = [2 ** i for i in range(0, hyperparams['nb_layers'])]
        self.in_channels = [in_channels] + [hyperparams['nb_filters'] for _ in range(hyperparams['nb_layers'])]
        self.dilated_causal_convs = nn.ModuleList(
            [DilatedCausalConv1d(hyperparams, self.dilation_factors[i], self.in_channels[i]) for i in
             range(hyperparams['nb_layers'])])
        for dilated_causal_conv in self.dilated_causal_convs:
            dilated_causal_conv.apply(weights_init)

        self.hidden_layer = nn.Conv1d(in_channels=self.in_channels[-1],
                                      out_channels=hyperparams['hidden_size'],
                                      kernel_size=1)
        self.hidden_layer.apply(weights_init)
        self.leaky_relu = nn.LeakyReLU(0.1)

        # intensity prediction
        # history
        self.w = nn.parameter.Parameter(torch.rand(num_types, hyperparams['hidden_size']))
        
        # future
        self.interpolator = interpolator
        
        # offset
        self.b = nn.parameter.Parameter(torch.ones(num_types))

        self.softplus = nn.Softplus()

    def init_param(self, *args):
        """
        Used for weight initialization, output ~ U(-1/hidden_size,1/hidden_size)
        """
        tmp = self.hidden_size
        return torch.rand(*args) * 2 / tmp - 1 / tmp
    
    def temporal_enc(self, time, non_pad_mask):
        result = time.unsqueeze(-1) / self.position_vec.to(time.device)
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask
    
    def temporal_enc_future(self, time, non_pad_mask):
        result = time.unsqueeze(-1) / self.position_vec.to(time.device)
        result[:, :, :, 0::2] = torch.sin(result[:, :, :, 0::2])
        result[:, :, :, 1::2] = torch.cos(result[:, :, :, 1::2])
        return result * non_pad_mask
    
    @staticmethod
    def get_non_pad_mask(seq):
        """ Get the non-padding positions. """

        assert seq.dim() == 2
        return seq.ne(0).type(torch.float).unsqueeze(-1)
    
    def forward(self, event_time, event_type):
        non_pad_mask = self.get_non_pad_mask(event_type)
        
        temporal_enc = self.temporal_enc(event_time, non_pad_mask)
        event_emb = self.event_emb(event_type)
        
        x = temporal_enc + event_emb
        
        x = x.transpose(1,2)

        for dilated_causal_conv in self.dilated_causal_convs:
            x = dilated_causal_conv(x)
        enc_output = self.leaky_relu(self.hidden_layer(x)).transpose(1,2) * non_pad_mask
        
        return enc_output # shape = (bs, L, hidden_size)
    
    def get_lambdas(
        self,
        event_time: torch.Tensor,
        enc_output: torch.Tensor,
        delta_t: torch.Tensor,
        non_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes lambdas on timestamps t_j
        
        args:
            event_time - torch.Tensor, event times with shape = (bs, L)
            enc_output - torch.Tensor, WaveNet model forward pass output
            delta_t - torch.Tensor, delta times since event_time where to compute the intensities, shape = (bs, L, n_times)
            non_pad_mask - torch.Tensor, indicates non pad ids
            
        returns:
            intens - torch.Tensor, intensity tensor, shape = (bs, num_types, L, n_times)
        """
        event_time = event_time.unsqueeze(2)
        
        # history
        history = self.w @ hidden # (bs, L, num_types)
        history = history.unsqueeze(2)
        
        # future        
        temp_enc = self.temporal_enc_future(delta_t, non_pad_mask.unsqueeze(-1)) # shape = (bs, L, n_times, hidden_size)
        future_input = hidden.unsqueeze(2) + temp_enc
        
        future = self.interpolator(future_input)
        

        # offset
        b = self.b[None,None, None, :]
        tmp = future + history + b
        out = self.softplus(tmp).transpose(2,3) # shape = (bs, L, num_types, n_times)
        return out
    