import torch
import torch.nn as nn

from .cotic_layers import ContinuousConv1D


class COTIC(nn.Module):
    """
    Continuous-Time Convolutional (COTIC) neural network module for processing continuous-time event sequences.

    This module is designed for modeling event sequences with continuous-time information
    using convolutional neural networks.

    Args:
    - in_channels (int): Number of input channels.
    - kernel_size (int): Size of the convolutional kernel.
    - nb_filters (int): Number of filters.
    - nb_layers (int): Number of convolutional layers.
    - num_types (int): Number of event types.

    Input Shape:
    Has beginning of stream event and zero padding
        - event_times (torch.Tensor): Shape = (batch_size, sequence_length),
          representing the continuous event arrival times.
        - event_types (torch.Tensor): Shape = (batch_size, sequence_length),
          representing event types (integers).

    Output Shape:
        - torch.Tensor: Shape = (batch_size, sequence_length, nb_filters),
          encoder output for the input event sequence.

    Examples:
    ```python
    # Create a COTIC model
    model = COTIC(in_channels=64, kernel_size=3, nb_filters=32, nb_layers=2, num_types=10)

    # Forward pass
    event_times = torch.tensor([[0, 0.1, 0.5, 1.2, 0], [0, 0.2, 0.7, 1.4, 0]])
    event_types = torch.tensor([[5, 1, 3, 2, 0], [5, 2, 4, 1, 0]])
    encoder_output = model(event_times, event_types)
    ```

    Note:
    - In the input `event_times`, event arrival times should be in chronological order.
    - The input `event_types` should represent event types as integers.
    """
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        nb_filters: int,
        nb_layers: int,
        num_types: int,
        dropout: float = 0.1
    ) -> None:
        """
        Initialize a COTIC (Continuous-Time Convolutional) neural network module.

        Args:
        - in_channels (int): Number of input channels.
        - kernel_size (int): Size of the convolutional kernel.
        - nb_filters (int): Number of filters.
        - nb_layers (int): Number of convolutional layers.
        - num_types (int): Number of event types.
        """
        super().__init__()
        self.event_emb = nn.Embedding(num_types + 1, in_channels, padding_idx=0)

        self.in_channels = [in_channels] + [nb_filters] * nb_layers
        self.dilation_factors = [2**i for i in range(0, nb_layers)]

        self.num_types = num_types
        self.nb_layers = nb_layers
        self.nb_filters = nb_filters

        self.continuous_convolutions = nn.ModuleList(
            [
                ContinuousConv1D(
                    kernel_size,
                    self.in_channels[i],
                    nb_filters,
                    self.dilation_factors[i]
                )
                for i in range(nb_layers)
            ]
        )

        self.dropouts = nn.ModuleList(
            [
                nn.Dropout(dropout)
                for i in range(nb_layers)
            ]
        )

    def forward(
        self, event_times: torch.Tensor, event_types: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass that computes continuous convolutions and returns encoder output.

        Args:
        - event_times (torch.Tensor): Shape = (batch_size, sequence_length),
          representing the continuous event arrival times.
        - event_types (torch.Tensor): Shape = (batch_size, sequence_length),
          representing event types (integers).

        Returns:
        - torch.Tensor: Shape = (batch_size, sequence_length, nb_filters),
          encoder output for the input event sequence.
        """
        enc_output = self.event_emb(event_types)

        for dropout, conv in zip(self.dropouts, self.continuous_convolutions):
            enc_output = dropout(
                torch.nn.functional.leaky_relu(
                    conv(event_times, enc_output), 0.1
                )
            )

        return enc_output
