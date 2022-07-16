import torch
import torch.nn as nn


class Kernel(nn.Module):
    def __init__(self, hidden1, hidden2, hidden3, in_channels, out_channels):
        super().__init__()
        self.layer_1 = nn.Linear(1, hidden1)
        self.relu_1 = nn.ReLU()
        self.layer_2 = nn.Linear(hidden1, hidden2)
        self.relu_2 = nn.ReLU()
        self.layer_3 = nn.Linear(hidden2, hidden3)
        self.relu_3 = nn.ReLU()
        self.layer_4 = nn.Linear(hidden3, in_channels * out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    def forward(self, x):
        x = self.relu_1(self.layer_1(x))
        x = self.relu_2(self.layer_2(x))
        x = self.relu_3(self.layer_3(x))
        x = self.layer_4(x)
        x = x.reshape(-1, self.in_channels, self.out_channels)
        
        return x
        

class ContConv1d(nn.Module):
    def __init__(self, kernel, kernel_size):
        super().__init__()
        self.kernel = kernel
        self.kernel_size = kernel_size
        
    def forward(self, times, features):
        S = times.unsqueeze(2).repeat(1,1,times.shape[1])
        S = torch.relu(S - S.transpose(1,2))*torch.triu(torch.ones((times.shape[1],times.shape[1])), diagonal = -self.kernel_size)
        mask = torch.tril(torch.ones((times.shape[1],times.shape[1])), diagonal = -1)*torch.triu(torch.ones((times.shape[1],times.shape[1])), diagonal = -self.kernel_size)
        mask = mask.bool()
        bs, L, _ = S.shape
        S = S.reshape(-1).unsqueeze(1)
        
        kernel_values = self.kernel(S)
        _, in_channels, out_channels = kernel_values.shape
        kernel_values = kernel_values.reshape(bs, L, L, in_channels, out_channels)
        kernel_values[:,~mask,...] = 0
        
        features = features[:,:,None,:,None]
        out = features * kernel_values
        out = out.sum(dim=(2,3))
        
        return out
    
class CCNN(nn.Module):
    def __init__(self, n_types, in_features, num_filters, kernel_sizes, kernel_params):
        super().__init__()
        self.emb = nn.Embedding(n_types, in_features)
        self.modules = nn.ModuleList([ContConv1d(Kernel(*kernel_params[i]), kernel_sizes[i])for i in range(num_filters)])
        
    def forward(self, times, features):
        x = self.emb(features)
        for module in self.modules:
            x = module(x)
            
        return x