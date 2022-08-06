import torch
import torch.nn as nn

class PredictionHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_types: int
    ) -> None:
        super().__init__()
        self.return_time_prediction = nn.Sequential(nn.Linear(in_channels, 128),nn.ReLU(),nn.Linear(128,1))
        self.event_type_prediction = nn.Sequential(nn.Linear(in_channels, 128),nn.ReLU(),nn.Linear(128,num_types))
        
    def forward(
        self,
        enc_output: torch.Tensor
    ) -> torch.Tensor:
        return self.return_time_prediction(enc_output), self.event_type_prediction(enc_output)
