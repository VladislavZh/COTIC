import torch
import torch.nn as nn

import math

class WaveNet(nn.Module):
    def __init__(self, hyperparams: dict, in_channels: int, num_types: int):
        super().__init__()

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight.data)
                init.zeros_(m.bias.data)

        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / in_channels) for i in range(in_channels)])
        
        self.emb = nn.Embedding(num_types + 1, in_channels, padding_idx=0)
        
        self.dilation_factors = [2 ** i for i in range(0, hyperparams['nb_layers'])]
        self.in_channels = [in_channels] + [hyperparams['nb_filters'] for _ in range(hyperparams['nb_layers'])]
        self.num_types = num_types
        self.dilated_causal_convs = nn.ModuleList(
            [DilatedCausalConv1d(hyperparams, self.dilation_factors[i], self.in_channels[i]) for i in
             range(hyperparams['nb_layers'])])
        for dilated_causal_conv in self.dilated_causal_convs:
            dilated_causal_conv.apply(weights_init)

        self.output_layer_type = nn.Conv1d(in_channels=self.in_channels[-1],
                                           out_channels=self.num_types,
                                           kernel_size=1)
        self.output_layer_time = nn.Conv1d(in_channels=self.in_channels[-1],
                                           out_channels=1,
                                           kernel_size=1)
        self.output_layer_type.apply(weights_init)
        self.output_layer_time.apply(weights_init)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def temporal_enc(self, time, non_pad_mask):
        result = time.unsqueeze(-1) / self.position_vec.to(time.device)
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
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
        
        for dilated_causal_conv in self.dilated_causal_convs:
            x = dilated_causal_conv(x)
        time_pred = self.leaky_relu(self.output_layer_time(x)) * non_pad_mask
        type_pred = self.leaky_relu(self.output_layer_type(x)) * non_pad_mask
        
        return time_pred, type_pred
