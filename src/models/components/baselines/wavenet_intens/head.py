import torch
import torch.nn as nn

class PredictionHead:
    def __init__(
        in_channels: int,
        num_types: int
    ) -> None:
        super().__init__()
        self.return_time_prediction = nn.Linear(in_channels, 1)
        self.event_type_prediction = nn.Linear(in_channels, num_types)
        
    def forward(
        self,
        enc_output: torch.Tensor
    ) -> torch.Tensor:
        return self.return_time_prediction(enc_output), self.event_type_prediction(enc_output)
