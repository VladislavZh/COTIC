import torch
import torch.nn as nn

from typing import Type, Tuple


class ContConv1dDense(nn.Module):
    """
    Continuous convolution layer
    """
    def __init__(
        self,
        kernel: Type[nn.Module],
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
        conv_mask = torch.tril(torch.ones(true_ids.shape[1],true_ids.shape[1]), diagonal=-1) * \
                    torch.triu(torch.ones(true_ids.shape[1],true_ids.shape[1]), diagonal=-(sim_size+1)*kernel_size)
        len_mask = torch.arange(true_ids.shape[1])[None,:,None].repeat(true_ids.shape[0],1,true_ids.shape[1])
        len_mask = (len_mask<=(sim_size + 1)*(lengths[:,None,None]-1))
        conv_mask = conv_mask.bool()
        dt_mask = dt_mask * conv_mask.unsqueeze(0) * len_mask

        delta_times = torch.zeros_like(S)
        delta_times[dt_mask] = S[dt_mask]
        return delta_times, dt_mask
        
    def forward(self, times, features, lengths, true_ids, sim_size):
        delta_times, dt_mask = self.__conv_matrix_constructor(times, lengths, true_ids, sim_size, self.kernel_size)
        bs, L, _ = delta_times.shape
        kernel_values = torch.zeros(bs,L,L,self.in_channels, self.out_channels)
        kernel_values[dt_mask,:,:] = self.kernel(delta_times[dt_mask].unsqueeze(1))
        
        features = features[:,None,:,:,None]
        out = features * kernel_values
        out = out.sum(dim=(2,3))
        
        return out

    
class SimCNN(nn.Module):
    """
    Aggregates simulated times outputs and returns tensor that is applicable for ContConv1dDense
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        sim_size: int
    ):
        super().__init__()
        
        assert kernel_size % 2 == 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sim_size = sim_size
        
        padding = (kernel_size - 1)//2
        in_channels_list = [in_channels] + [out_channels] * (sim_size - 1)
        out_channels_list = [out_channels] * sim_size
        self.convs = nn.Sequential(*([nn.Sequential(nn.Conv1d(in_channels_list[i], out_channels_list[i], self.kernel_size, padding = padding),
                                             nn.ReLU(),
                                             nn.MaxPool1d(2, stride = 1)) for i in range(sim_size-1)] +\
                                   [nn.Sequential(nn.Conv1d(in_channels_list[sim_size-1], out_channels_list[sim_size-1], self.kernel_size, padding = padding),
                                             nn.MaxPool1d(2, stride = 1))]))
        
    def __add_sim_out(
        self,
        out: torch.Tensor,
        true_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Adds the beginning of stream all ones vector and simulated times all zeros vectors to the output
        """
        bs, L, out_channels = out.shape
        true_ids = true_ids.unsqueeze(2).repeat(1,1,out_channels)
        
        # sim events
        sim_events = torch.zeros(bs, L, self.sim_size, out_channels)
        out = out.unsqueeze(2)
        out = torch.concat([sim_events, out], dim = 2)
        out = out.reshape(bs, L*(self.sim_size + 1), out_channels)
        
        # beginning of stream event
        bos = torch.ones(bs, 1, out_channels)
        out = torch.concat([bos, out], dim = 1)
        
        # masking
        out[~true_ids] = 0
        return out
        
    
    def forward(
        self,
        times: torch.Tensor,
        features: torch.Tensor,
        lengths: torch.Tensor,
        true_ids: torch.Tensor
    ) -> torch.Tensor:
        delta_times = times[:,1:] - times[:,:-1]
        z = torch.arange(0,self.in_channels)[None,None,:].repeat(1,delta_times.shape[1],1)
        z = torch.cos(
                delta_times.unsqueeze(2)/10000**((z - 1)/self.in_channels)
            ) * (z % 2 == 1) \
            + torch.sin(
                delta_times.unsqueeze(2)/10000**(z/self.in_channels)
            ) * (z % 2 == 0)
        X = features[:,1:,:] + z
        bs, L, _ = X.shape
        X = X.reshape(bs * L//(self.sim_size + 1), (self.sim_size + 1), self.in_channels).transpose(1,2)
        X = self.convs(X)
            
        X = X.transpose(1,2)
        X = X.reshape(bs, L//(self.sim_size + 1), self.out_channels)
        return self.__add_sim_out(X, true_ids)
