import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

from typing import Tuple


class ContConv1d(nn.Module):
    """
    Continuous convolution layer for true events
    """
    def __init__(
        self,
        kernel: nn.Module,
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
        dropout=0.1,
        include_zero_lag: bool = False,
        skip_connection: bool = False
    ):
        """
        args:
            kernel - torch.nn.Module, Kernel neural net that takes (*,1) as input and returns (*, in_channles, out_channels) as output
            kernel_size - int, convolution layer kernel size
            in_channels - int, features input size
            out_channles - int, output size
            dilation - int, convolutional layer dilation (default = 1)
            include_zero_lag - bool, indicates if the model should use current time step features for prediction
            skip_connection - bool, indicates if the model should add skip connection in the end, in_channels == out_channels
        """
        super().__init__()
        assert dilation >= 1
        assert in_channels >= 1
        assert out_channels >= 1
        
        if skip_connection:
            assert in_channels == out_channels
        
        self.kernel = kernel
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.include_zero_lag = include_zero_lag
        self.skip_connection = skip_connection
        
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / self.in_channels) for i in range(self.in_channels)])
        
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def __temporal_enc(self, time, non_pad_mask):
        result = time.unsqueeze(-1) / self.position_vec.to(time.device)
        result[..., 0::2] = torch.sin(result[..., 0::2])
        result[..., 1::2] = torch.cos(result[..., 1::2])
        return result * non_pad_mask
        
    @staticmethod
    def __conv_matrix_constructor(
        times: torch.Tensor,
        features: torch.Tensor,
        non_pad_mask: torch.Tensor,
        kernel_size: int,
        dilation: int,
        include_zero_lag: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns delta_times t_i - t_j, where t_j are true events and the number of delta_times per row is kernel_size
        
        args:
            times - torch.Tensor of shape = (bs, max_len), Tensor of all times
            features - torch.Tensor of shape = (bs,max_len, in_channels), input tensor
            non_pad_mask - torch.Tensor of shape = (bs, max_len),  indicates non_pad timestamps
            kernel_size - int, covolution kernel size
            dilation - int, convolution dilation
            include_zero_lag: bool, indicates if we should use zero-lag timestamp
        
        returns:
            delta_times - torch.Tensor of shape = (bs, kernel_size, max_len) with delta times value between current time and kernel_size true times before it
            pre_conv_features - torch.Tensor of shape = (bs, kernel_size, max_len, in_channels) with corresponding input features of timestamps in delta_times
            dt_mask - torch.Tensor of shape = (bs, kernel_size, max_len), bool tensor that indicates delta_times true values
        """
        # parameters
        padding = (kernel_size - 1) * dilation if include_zero_lag else kernel_size * dilation
        kernel = torch.eye(kernel_size).unsqueeze(1).to(times.device)
        in_channels = features.shape[2]
        
        # convolutions
        pre_conv_times = F.conv1d(times.unsqueeze(1), kernel, padding = padding, dilation = dilation)
        pre_conv_features = F.conv1d(features.transpose(1,2), kernel.repeat(in_channels,1,1), padding = padding, dilation = dilation, groups=in_channels)
        dt_mask = F.conv1d(non_pad_mask.float().unsqueeze(1), kernel.float(), padding = padding, dilation = dilation).long().bool()
        
        # deleting extra values
        pre_conv_times = pre_conv_times[:,:,:-(padding + dilation * (1 - int(include_zero_lag)))]
        pre_conv_features = pre_conv_features[:,:,:-(padding + dilation * (1 - int(include_zero_lag)))]
        dt_mask = dt_mask[:,:,:-(padding + dilation * (1 - int(include_zero_lag)))] * non_pad_mask.unsqueeze(1)
        
        # updating shape
        bs, L, dim = features.shape
        pre_conv_features = pre_conv_features.reshape(bs, dim, kernel_size, L)
        
        # computing delte_time and deleting masked values
        delta_times = times.unsqueeze(1) - pre_conv_times
        delta_times[~dt_mask] = 0
        pre_conv_features = torch.permute(pre_conv_features, (0, 2, 3, 1))
        pre_conv_features[~dt_mask,:] = 0
        
        return delta_times, pre_conv_features, dt_mask
        
    def forward(self, times, features, non_pad_mask):
        """
        Neural net layer forward pass
        
        args:
            times - torch.Tensor, shape = (bs, L), event times
            features - torch.Tensor, shape = (bs, L, in_channels), event features
            non_pad_mask - torch.Tensor, shape = (bs,L), mask that indicates non pad values
            
        returns:
            out - torch.Tensor, shape = (bs, L, out_channels)
        """
        delta_times, features_kern, dt_mask = self.__conv_matrix_constructor(times, features, non_pad_mask, self.kernel_size, self.dilation, self.include_zero_lag)
        bs, k, L = delta_times.shape
        kernel_values = torch.zeros(bs,k,L,self.in_channels, self.out_channels).to(times.device)
        kernel_values[dt_mask,:,:] = self.kernel(self.__temporal_enc(delta_times[dt_mask], non_pad_mask))
        out = features_kern.unsqueeze(-1) * kernel_values
        out = out.sum(dim=(1,3))
        if self.skip_connection:
            out = out + features
        #out = self.dropout(self.norm(out))
        out = self.norm(out)
        return out


class ContConv1dSim(nn.Module):
    """
    Continuous convolution layer
    """
    def __init__(
        self,
        kernel: nn.Module,
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        dropout=0.1
    ):
        """
        args:
            kernel - torch.nn.Module, Kernel neural net that takes (*,1) as input and returns (*, in_channles, out_channels) as output
            kernel_size - int, convolution layer kernel size
            in_channels - int, features input size
            out_channles - int, output size
        """
        super().__init__()
        self.kernel = kernel
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / self.in_channels) for i in range(self.in_channels)])
        
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def __temporal_enc(self, time, non_pad_mask):
        result = time.unsqueeze(-1) / self.position_vec.to(time.device)
        result[..., 0::2] = torch.sin(result[..., 0::2])
        result[..., 1::2] = torch.cos(result[..., 1::2])
        return result * non_pad_mask
        
    @staticmethod
    def __conv_matrix_constructor(
        times: torch.Tensor,
        true_times: torch.Tensor,
        true_features: torch.Tensor,
        non_pad_mask: torch.Tensor,
        kernel_size: int,
        sim_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns delta_times t_i - t_j, where t_j are true events and the number of delta_times per row is kernel_size
        
        args:
            times - torch.Tensor of shape = (bs, (sim_size+1)*(max_len-1)+1), Tensor of all times
            true_times - torch.Tensor of shape = (bs, max_len), Tensor of true times
            true_features - torch.Tensor of shape = (bs, max_len, in_channels), input tensor
            non_pad_mask - torch.Tensor of shape = (bs, max_len),  indicates non_pad timestamps
            kernel_size - int, covolution kernel size
            sim_size - int, simulated times size
        
        returns:
            delta_times - torch.Tensor of shape = (bs, kernel_size, (sim_size+1)*(max_len-1)+1)) with delta times value between current time and kernel_size true times before it
            pre_conv_features - torch.Tensor of shape = (bs, kernel_size, (sim_size+1)*(max_len-1)+1), in_channels) with corresponding input features of timestamps in delta_times
            dt_mask - torch.Tensor of shape = (bs, kernel_size, (sim_size+1)*(max_len-1)+1)), bool tensor that indicates delta_times true values
        """
        # parameters
        padding = (kernel_size - 1) * 1
        kernel = torch.eye(kernel_size).unsqueeze(1).to(times.device)
        in_channels = true_features.shape[2]
        
        # true values convolutions
        pre_conv_times = F.conv1d(true_times.unsqueeze(1), kernel, padding = padding, dilation = 1)
        pre_conv_features = F.conv1d(true_features.transpose(1,2), kernel.repeat(in_channels,1,1), padding = padding, dilation = 1, groups=true_features.shape[2])
        dt_mask = F.conv1d(non_pad_mask.float().unsqueeze(1), kernel.float(), padding = padding, dilation = 1).long().bool()
        
        # deleting extra values
        if padding>0:
            pre_conv_times = pre_conv_times[:,:,:-padding]
            pre_conv_features = pre_conv_features[:,:,:-padding]
            dt_mask = dt_mask[:,:,:-padding] * non_pad_mask.unsqueeze(1)
        else:
            dt_mask = dt_mask * non_pad_mask.unsqueeze(1)
        
        # reshaping features output
        bs, L, dim = true_features.shape
        pre_conv_features = pre_conv_features.reshape(bs, dim, kernel_size, L)
        
        # adding sim_times
        pre_conv_times = pre_conv_times.unsqueeze(-1).repeat(1,1,1,sim_size+1)
        pre_conv_times = pre_conv_times.flatten(2)
        if sim_size>0:
            pre_conv_times = pre_conv_times[...,:-sim_size]
        
        pre_conv_features = pre_conv_features.unsqueeze(-1).repeat(1,1,1,1,sim_size+1)
        pre_conv_features = pre_conv_features.flatten(3)
        if sim_size>0:
            pre_conv_features = pre_conv_features[...,:-sim_size]
        
        dt_mask = dt_mask.unsqueeze(-1).repeat(1,1,1,sim_size+1)
        dt_mask = dt_mask.flatten(2)
        dt_mask = dt_mask[...,sim_size:]
        
        delta_times = times.unsqueeze(1) - pre_conv_times
        delta_times[~dt_mask] = 0
        
        pre_conv_features = torch.permute(pre_conv_features, (0, 2, 3, 1))
        pre_conv_features[~dt_mask,:] = 0
        return delta_times, pre_conv_features, dt_mask
        
    def forward(self, times, true_times, true_features, non_pad_mask, sim_size):
        """
        Neural net layer forward pass
        
        args:
            times - torch.Tensor of shape = (bs, (sim_size+1)*(max_len-1)+1), Tensor of all times
            true_times - torch.Tensor of shape = (bs, max_len), Tensor of true times
            true_features - torch.Tensor of shape = (bs, max_len, in_channels), input tensor
            non_pad_mask - torch.Tensor of shape = (bs, max_len),  indicates non_pad timestamps\
            sim_size - int, simulated times size
            
        returns:
            out - torch.Tensor, shape = (bs, L, out_channels)
        """
        delta_times, features_kern, dt_mask = self.__conv_matrix_constructor(times, true_times, true_features, non_pad_mask, self.kernel_size, sim_size)

        bs, k, L = delta_times.shape
        kernel_values = torch.zeros(bs,k,L,self.in_channels, self.out_channels).to(times.device)
        kernel_values[dt_mask,:,:] = self.kernel(self.__temporal_enc(delta_times[dt_mask], non_pad_mask))
        out = features_kern.unsqueeze(-1) * kernel_values
        out = out.sum(dim=(1,3))
        
        # out = self.dropout(self.norm(out))
        
        return out
