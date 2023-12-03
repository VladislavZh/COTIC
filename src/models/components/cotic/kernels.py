import torch.nn as nn


class Kernel(nn.Module):
    def __init__(self, hidden1, hidden2, hidden3, in_channels, out_channels):
        super().__init__()
        self.args = [hidden1, hidden2, hidden3, in_channels, out_channels]

        self.layers = nn.Sequential(
            nn.Linear(in_channels, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(inplace=True),
            nn.Linear(hidden3, in_channels * out_channels)
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Apply He initialization to the linear layers
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        shape = list(x.shape)[:-1] + [self.in_channels, self.out_channels]
        x = self.layers(x)
        x = x.reshape(*shape)

        return x
