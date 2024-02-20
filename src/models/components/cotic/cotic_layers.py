from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class ContinuousConvolutionBase(nn.Module, ABC):
    """
    Base class for continuous convolutional layers, providing core functionalities
    for different types of continuous convolutions without implementing a forward pass.

    This class serves as a foundation for specific continuous convolution implementations.
    It handles the initial setup and common operations but requires subclasses to define
    their own forward pass logic.

    Attributes:
    - kernel_network (nn.Module): A neural network module that serves as the convolution kernel.
      It takes input of shape (*,1) and outputs a tensor of shape (*, input_channels, output_channels).
    - kernel_size (int): The size of the convolutional kernel.
    - input_channels (int): The number of input channels.
    - output_channels (int): The number of output channels.
    - dilation (int): The dilation factor of the convolutional layer.

    Methods:
    - construct_conv_matrix: Constructs and returns matrices necessary for the convolution operation.
    """

    def __init__(
            self,
            kernel_network: nn.Module,
            kernel_size: int,
            input_channels: int,
            output_channels: int,
            dilation: int = 1
    ):
        """
        Initialize the ContinuousConvolutionBase layer.

        Args:
        - kernel_network (nn.Module): Kernel neural network that takes (*,1) as input and
                                      returns (*, input_channels, output_channels) as output.
        - kernel_size (int): Convolution layer kernel size.
        - input_channels (int): Input feature size.
        - output_channels (int): Output feature size.
        - dilation (int, optional): Convolutional layer dilation (default=1).
        """
        super().__init__()
        assert dilation >= 1
        assert input_channels >= 1
        assert output_channels >= 1

        self.kernel_network = kernel_network
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.skip_connection = nn.Conv1d(
            in_channels=input_channels, out_channels=output_channels, kernel_size=1
        )

        self.leaky_relu = nn.LeakyReLU(0.1)

        self.layer_norm = nn.LayerNorm(output_channels)
        self.data_preparation_kernel = nn.Parameter(torch.eye(kernel_size).unsqueeze(1), requires_grad=False)
        self.data_preparation_padding = (self.kernel_size - 1) * self.dilation
        self.data_preparation_offset = 0

    def construct_conv_matrix(
            self,
            times: torch.Tensor,
            features: torch.Tensor,
            non_pad_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns delta_times t_i - t_j, where t_j are true events, and the number of delta_times per row is kernel_size.

        Args:
        - times (torch.Tensor): Tensor of shape=(batch_size, max_len), containing timestamps.
        - features (torch.Tensor): Tensor of shape=(batch_size, max_len, input_channels), input tensor.
        - non_pad_mask (torch.Tensor): Tensor of shape=(batch_size, max_len), indicating non-pad timestamps.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing delta_times, pre_conv_features, and dt_mask.
        """
        # Compute the number of input channels
        input_channels = features.shape[2]

        # Convolution operations
        pre_conv_times = F.conv1d(
            times.unsqueeze(1),
            self.data_preparation_kernel,
            padding=self.data_preparation_padding,
            dilation=self.dilation
        )
        pre_conv_features = F.conv1d(
            features.transpose(1, 2),
            self.data_preparation_kernel.repeat(input_channels, 1, 1),
            padding=self.data_preparation_padding,
            dilation=self.dilation,
            groups=input_channels,
        )
        dt_mask = (
            F.conv1d(
                non_pad_mask.float().unsqueeze(1),
                self.data_preparation_kernel.float(),
                padding=self.data_preparation_padding,
                dilation=self.dilation,
            )
            .bool()
        )

        # Remove extra values introduced by convolution padding
        pre_conv_times = pre_conv_times[:, :, :-(self.data_preparation_padding + self.data_preparation_offset)]
        pre_conv_features = pre_conv_features[:, :, :-(self.data_preparation_padding + self.data_preparation_offset)]
        dt_mask = dt_mask[:, :,
                  :-(self.data_preparation_padding + self.data_preparation_offset)] * non_pad_mask.unsqueeze(1)

        # Reshape pre_conv_features
        batch_size, seq_len, input_dim = features.shape
        pre_conv_features = pre_conv_features.reshape(batch_size, input_dim, self.kernel_size, seq_len)

        # Compute delta_times and mask out values according to the mask
        delta_times = times.unsqueeze(1) - pre_conv_times
        delta_times[~dt_mask] = 0
        pre_conv_features = torch.permute(pre_conv_features, (0, 2, 3, 1))
        pre_conv_features[~dt_mask, :] = 0

        return delta_times, pre_conv_features, dt_mask


class ContinuousConv1D(ContinuousConvolutionBase):
    def forward(
            self,
            times: torch.Tensor,
            features: torch.Tensor,
            non_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Conducts a continuous convolution operation on the input data_utils.

        This method applies the continuous convolution to the input features based on the given
        event times and a non-padding mask. It leverages a kernel network to compute the convolution
        kernel values and combines them with the input features to produce the output tensor.

        Args:
            times (torch.Tensor): A tensor of shape (batch_size, seq_len) containing the event times.
                                  Each entry in this tensor represents a timestamp for the corresponding
                                  event in the sequence.
            features (torch.Tensor): A tensor of shape (batch_size, seq_len, input_channels) containing
                                     the features associated with each event. These features are the
                                     inputs to the convolution operation.
            non_pad_mask (torch.Tensor): A boolean tensor of shape (batch_size, seq_len) indicating
                                         the non-padding elements in the sequence. This mask helps
                                         in differentiating actual data_utils from padded data_utils in operations.

        Returns:
            torch.Tensor: The output of the continuous convolution operation. It is a tensor of shape
                          (batch_size, seq_len, output_channels) where each element in the sequence
                          has been convolved with the kernel function, considering the continuous nature
                          of the data_utils.
        """

        delta_times, features_kern, dt_mask = self.construct_conv_matrix(
            times,
            features,
            non_pad_mask
        )
        delta_times /= self.dilation

        delta_times = delta_times.unsqueeze(-1)

        kernel_values = self.kernel_network(delta_times)
        kernel_values[~dt_mask, ...] = 0

        out = features_kern.unsqueeze(-1) * kernel_values
        out = out.sum(dim=(1, 3))

        out += self.skip_connection(features.transpose(1, 2)).transpose(1, 2)
        out = self.layer_norm(out)
        return out


class ContinuousConv1DSim(ContinuousConvolutionBase):
    """
    Continuous convolution layer with simulated times
    """

    def __init__(
            self,
            kernel_size: int,
            input_channels: int,
            output_channels: int
    ):
        """
        Initialize the ContinuousConv1DSim layer.

        Args:
        - kernel_size (int): Convolution layer kernel size.
        - input_channels (int): Input feature size.
        - output_channels (int): Output feature size.
        """
        kernel_network = nn.Linear(input_channels, output_channels, bias=False)

        super().__init__(
            kernel_network=kernel_network,
            kernel_size=kernel_size + 1,
            input_channels=input_channels,
            output_channels=output_channels,
            dilation=1
        )
        self.bias = nn.Parameter(torch.full(size=(input_channels, output_channels), fill_value=1 / output_channels))

    def forward(
            self,
            times: torch.Tensor,
            features: torch.Tensor,
            non_pad_mask: torch.Tensor,
            uniform_sample: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the neural network layer.

        Args:
        - times (torch.Tensor): Tensor of shape=(batch_size, seq_len), containing event times.
        - features (torch.Tensor): Tensor of shape=(batch_size, seq_len, input_channels), containing event features.
        - non_pad_mask (torch.Tensor): Tensor of shape=(batch_size, seq_len), mask indicating non-pad values.
        - uniform_sample (torch.Tensor): Tensor of shape=(sim_size), auxiliary tensor for output between times computation

        Returns:
        - torch.Tensor: Output tensor of shape=(batch_size, (sim_size+1)*(seq_len-1)+1, output_channels).
        """

        assert torch.all(uniform_sample == uniform_sample.sort().values)

        modified_features = self.kernel_network(features)
        features_bias = features @ self.bias  # shape = (bs, seq_len, out_channels)

        all_features = torch.concat([modified_features, features_bias], dim=-1)

        delta_times, features_kern, dt_mask = self.construct_conv_matrix(
            times,
            all_features,
            non_pad_mask
        )
        features_kern_linear = features_kern[..., :self.output_channels]
        features_kern_bias = features_kern[..., self.output_channels:]

        uniform_delta_t_scale = delta_times[:, -2, 1:]  # equivalent to the times[1:] - times[:-1]

        real_values = torch.sum(
            delta_times[:, :-1, ...].unsqueeze(-1) * features_kern_linear[:, :-1, ...] +
            features_kern_bias[:, :-1, ...],
            dim=1
        )  # shape = (bs, seq_len, out_channels)
        sim_values = torch.sum(
            delta_times[:, 1:, :-1].unsqueeze(-1) * features_kern_linear[:, 1:, :-1, ...] +
            features_kern_bias[:, 1:, :-1],
            dim=1
        )  # shape = (bs, seq_len - 1, out_channels)
        sim_values = (
                sim_values[:, :, None, :] +
                torch.sum(
                    uniform_delta_t_scale[:, None, :, None, None] *
                    uniform_sample[None, None, None, :, None] * features_kern_linear[:, 1:, :-1, None, :],
                    dim=1
                )
        )

        values_before_sim = real_values[:, :-1, :].unsqueeze(2)
        values_after_sim = real_values[:, -1:, :]

        bs, _, out_channels = values_after_sim.shape

        out = torch.cat([values_before_sim, sim_values], dim=2).reshape(bs, -1, out_channels)
        out = torch.cat([out, values_after_sim], dim=1)

        return out
