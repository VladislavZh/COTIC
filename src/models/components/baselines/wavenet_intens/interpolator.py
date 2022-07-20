import torch
import torch.nn as nn

class IntensPredictor:
    def __init__(
        self,
        in_channels: int,
        hidden1: int,
        hidden2: int,
        hidden3: int,
        num_types: int
    ):
        super().__init__()
        self.layer_1 = nn.Linear(in_channels, hidden1)
        self.relu_1 = nn.ReLU()
        self.layer_2 = nn.Linear(hidden1, hidden2)
        self.relu_2 = nn.ReLU()
        self.layer_3 = nn.Linear(hidden2, hidden3)
        self.relu_3 = nn.ReLU()
        self.layer_4 = nn.Linear(hidden3, num_types)
        
    def forward(self, x):
        x = self.relu_1(self.layer_1(x)) + x
        x = self.relu_2(self.layer_2(x)) + x
        x = self.relu_3(self.layer_3(x)) + x
        x = self.layer_4(x) + x
        
        return x
