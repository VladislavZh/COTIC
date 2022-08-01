import torch
import torch.nn as nn

from typing import Tuple


class ContConv1dDense(nn.Module):
    """
    Continuous convolution layer for true events
    """
    def __init__(
        self,
        kernel: nn.Module,
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        include_zero_lag: bool = False,
        skip_connection: bool = False
    ):
        super().__init__()
        self.kernel = kernel
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.include_zero_lag = include_zero_lag
        self.skip_connection = skip_connection
        
    @staticmethod
    def __conv_matrix_constructor(
        times: torch.Tensor,
        lengths: torch.Tensor,
        kernel_size: int,
        include_zero_lag: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns delta_times t_i - t_j, where t_j are true events and the number of delta_times per row is kernel_size
        
        args:
            times - torch.Tensor of shape = (bs, max_len), Tensor of all times
            lengths - torch.Tensor of shape = (bs,), Tensor of all true event lenghts (including bos event)
            kernel_size - int, covolution kernel size
            include_zero_lag: bool, indicates if we should ignore zero-lag timestamp
        
        returns:
            delta_times - torch.Tensor of shape = (bs, max_len, max_len) with delta times value between current time and kernel_size true times before it
            dt_mask - torch.Tensor of shape = (bs, max_len, max_len), bool tensor that indicates delta_times true values
        """
        S = times.unsqueeze(2).repeat(1,1,times.shape[1])
        S = S - S.transpose(1,2)
        
        true_ids = torch.arange(times.shape[1])[None,:].repeat(times.shape[0], 1).to(times.device)
        true_ids = (true_ids < lengths[:, None])
        dt_mask = true_ids.unsqueeze(1).repeat(1,true_ids.shape[1],1)
        
        if include_zero_lag == False:
            conv_mask = torch.tril(torch.ones(true_ids.shape[1],true_ids.shape[1]), diagonal=-1) * \
                        torch.triu(torch.ones(true_ids.shape[1],true_ids.shape[1]), diagonal=-kernel_size)
        else:
            conv_mask = torch.tril(torch.ones(true_ids.shape[1],true_ids.shape[1])) * \
                        torch.triu(torch.ones(true_ids.shape[1],true_ids.shape[1]), diagonal=(-kernel_size + 1))
        len_mask = torch.arange(times.shape[1])[None,:,None].repeat(times.shape[0],1,times.shape[1]).to(times.device)
        len_mask = (len_mask<=(lengths[:,None,None]-1))
        conv_mask = conv_mask.bool()
        dt_mask = dt_mask * conv_mask.unsqueeze(0).to(times.device) * len_mask

        delta_times = torch.zeros_like(S)
        delta_times[dt_mask] = S[dt_mask]
        return delta_times, dt_mask
        
    def forward(self, times, features, lengths):
        delta_times, dt_mask = self.__conv_matrix_constructor(times, lengths, self.kernel_size, self.include_zero_lag)
        bs, L, _ = delta_times.shape
        kernel_values = torch.zeros(bs,L,L,self.in_channels, self.out_channels).to(times.device)
        kernel_values[dt_mask,:,:] = self.kernel(delta_times[dt_mask].unsqueeze(1))
        features_unsq = features[:,None,:,:,None]
        out = features_unsq * kernel_values
        out = out.sum(dim=(2,3))
        if self.skip_connection:
            out = out + features
        return out

class ContConv1dDenseSim(nn.Module):
    """
    Continuous convolution layer
    """
    def __init__(
        self,
        kernel: nn.Module,
        kernel_size: int,
        in_channels: int,
        out_channels: int
    ):
        super().__init__()
        self.kernel = kernel
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    @staticmethod
    def __conv_matrix_constructor(
        times: torch.Tensor,
        lengths: torch.Tensor,
        true_ids: torch.Tensor,
        sim_size: int,
        kernel_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns delta_times t_i - t_j, where t_j are true events and the number of delta_times per row is kernel_size
        
        args:
            times - torch.Tensor of shape = (bs, max_len), Tensor of all times
            lengths - torch.Tensor of shape = (bs,), Tensor of all true event lenghts (including bos event)
            true_ids - torch.Tensor of shape = (bs, max_len), bool tensor that indicates true events
            sim_size - int, simulated times size
            kernel_size - int, covolution kernel size
        
        returns:
            delta_times - torch.Tensor of shape = (bs, max_len, max_len) with delta times value between current time and kernel_size true times before it
            dt_mask - torch.Tensor of shape = (bs, max_len, max_len), bool tensor that indicates delta_times true values
        """
        S = times.unsqueeze(2).repeat(1,1,times.shape[1])
        S = S - S.transpose(1,2)

        dt_mask = true_ids.unsqueeze(1).repeat(1,true_ids.shape[1],1)
        conv_mask = torch.tril(torch.ones(true_ids.shape[1],true_ids.shape[1])) * \
                    torch.triu(torch.ones(true_ids.shape[1],true_ids.shape[1]), diagonal=(-(sim_size+1)*kernel_size+1))
        len_mask = torch.arange(true_ids.shape[1])[None,:,None].repeat(true_ids.shape[0],1,true_ids.shape[1]).to(times.device)
        len_mask = (len_mask<=(sim_size + 1)*(lengths[:,None,None]-1))
        conv_mask = conv_mask.bool().to(times.device)
        dt_mask = dt_mask * conv_mask.unsqueeze(0) * len_mask

        delta_times = torch.zeros_like(S)
        delta_times[dt_mask] = S[dt_mask]
        return delta_times, dt_mask
        
    def forward(self, times, features, lengths, true_ids, sim_size):
        delta_times, dt_mask = self.__conv_matrix_constructor(times, lengths, true_ids, sim_size, self.kernel_size)
        bs, L, _ = delta_times.shape
        kernel_values = torch.zeros(bs,L,L,self.in_channels, self.out_channels).to(times.device)
        kernel_values[dt_mask,:,:] = self.kernel(delta_times[dt_mask].unsqueeze(1))
        
        features = features[:,None,:,:,None]
        out = features * kernel_values
        out = out.sum(dim=(2,3))
        
        return out
