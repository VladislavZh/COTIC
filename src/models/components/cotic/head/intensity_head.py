import torch.nn as nn

from src.models.components.cotic.cotic_layers import ContinuousConv1DSim
from src.models.components.cotic.head.head_core import IntensityHead


class LinearKernelIntensityHead(IntensityHead):
    def __init__(
        self,
        kernel_size: int,
        nb_filters: int,
        mlp_layers: int,
        num_types: int
    ) -> None:
        self.convolution = ContinuousConv1DSim(
            kernel_size, nb_filters, nb_filters
        ),
        self.activation = nn.LeakyReLU(0.1)
        self.layer1 = nn.Linear(nb_filters, num_types),
                nn.Softplus(100),
            ]
        )