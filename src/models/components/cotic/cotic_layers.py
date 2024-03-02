from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class ContinuousConvolutionBase(nn.Module, ABC):
    """
    Base class for continuous convolutional layers. It provides the core functionalities
    for different types of continuous convolutions but does not implement a forward pass,
    which should be defined in subclasses.

    This class sets up the initial parameters and methods that are common across different
    continuous convolution implementations, such as constructing convolution matrices.

    Attributes:
        kernel_size (int): The size of the convolutional kernel.
        input_channels (int): The number of input channels in the input feature map.
        output_channels (int): The desired number of output channels in the output feature map.
        dilation (int): The dilation factor for the convolution operation, increasing the spatial
                        reach of the kernel by introducing gaps between the kernel elements.
    """

    def __init__(
            self,
            kernel_size: int,
            input_channels: int,
            output_channels: int,
            dilation: int = 1
    ):
        """
        Initializes the ContinuousConvolutionBase layer with essential configurations.

        Parameters:
            kernel_size (int): Size of the convolution kernel.
            input_channels (int): Number of channels in the input features.
            output_channels (int): Number of channels in the output features.
            dilation (int, optional): Dilation factor of the convolution, defaults to 1.
        """
        super().__init__()
        assert dilation >= 1
        assert input_channels >= 1
        assert output_channels >= 1

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.data_preparation_kernel = nn.Parameter(torch.eye(kernel_size).unsqueeze(1).repeat(2 * output_channels + 1, 1, 1), requires_grad=False)
        self.data_preparation_padding = (self.kernel_size - 1) * self.dilation

    def construct_conv_matrix(
            self,
            times: torch.Tensor,
            features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs matrices necessary for the convolution operation, including time deltas
        and pre-convolution feature matrices.

        Parameters:
            times (torch.Tensor): A tensor of shape (batch_size, seq_len) containing timestamps for each event.
            features (torch.Tensor): A tensor of shape (batch_size, seq_len, input_channels) containing features
                                     associated with each event.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - delta_times: A tensor representing the time differences between events, used to weight the convolution.
                - pre_conv_features: A tensor of pre-convolved features, prepared for the convolution operation.
        """
        all_values = torch.concat(
            [times.unsqueeze(-1), features],
            dim=-1
        )

        pre_conv_all_values = F.conv1d(
            all_values.transpose(1, 2),
            self.data_preparation_kernel,
            padding=self.data_preparation_padding,
            dilation=self.dilation,
            groups=2 * self.output_channels + 1,
        )

        # Remove extra values introduced by convolution padding
        pre_conv_all_values = pre_conv_all_values[:, :, :-self.data_preparation_padding]
        pre_conv_all_values = pre_conv_all_values.reshape(times.shape[0], 2 * self.output_channels + 1, self.kernel_size, times.shape[1])

        pre_conv_times = pre_conv_all_values[:, 0, :, :]
        pre_conv_features = pre_conv_all_values[:, 1:, :, :]

        # Compute delta_times and mask out values according to the mask
        delta_times = times.unsqueeze(1) - pre_conv_times
        pre_conv_features = torch.permute(pre_conv_features, (0, 2, 3, 1))

        return delta_times, pre_conv_features


class ContinuousConv1D(ContinuousConvolutionBase):
    """
    Implements a continuous convolution layer suitable for processing sequences of events with continuous time stamps,
    extending the ContinuousConvolutionBase with a specific forward pass strategy.

    This layer applies a convolution operation that considers the timing of events, making it suitable for time-series
    data where the temporal aspect of the data is significant.

    Inherits from ContinuousConvolutionBase.
    """
    def __init__(
            self,
            kernel_size: int,
            input_channels: int,
            output_channels: int,
            dilation: int
    ):
        """
        Initializes the ContinuousConv1D layer with the specified kernel size, input/output channels, and dilation factor.

        Parameters:
            kernel_size (int): The size of the convolution kernel.
            input_channels (int): The number of input channels.
            output_channels (int): The number of output channels.
            dilation (int): The dilation factor for the convolution operation.
        """

        super().__init__(
            kernel_size=kernel_size,
            input_channels=input_channels,
            output_channels=output_channels,
            dilation=dilation
        )
        self.kernel_network_weight = nn.Linear(input_channels, output_channels, bias=False)
        self.kernel_network_bias = nn.Parameter(torch.full(size=(input_channels, output_channels), fill_value=1 / output_channels))

        self.skip_connection = nn.Linear(
            input_channels, output_channels
        )

        self.layer_norm = nn.LayerNorm(output_channels)

    def forward(
            self,
            times: torch.Tensor,
            features: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs the forward pass of the ContinuousConv1D layer.

        Parameters:
            times (torch.Tensor): A tensor of shape (batch_size, seq_len) containing the times of each event.
            features (torch.Tensor): A tensor of shape (batch_size, seq_len, input_channels) containing the features
                                     associated with each event.

        Returns:
            torch.Tensor: The output of the layer after applying the continuous convolution operation, skip connection,
                          and layer normalization. The output tensor has a shape (batch_size, seq_len, output_channels).
        """

        modified_features = self.kernel_network_weight(features)
        features_bias = features @ self.kernel_network_bias  # shape = (bs, seq_len, out_channels)

        all_features = torch.concat([modified_features, features_bias], dim=-1)

        delta_times, features_kern = self.construct_conv_matrix(
            times,
            all_features
        )
        delta_times /= self.dilation
        features_kern_linear = features_kern[..., :self.output_channels]
        features_kern_bias = features_kern[..., self.output_channels:]

        out = torch.sum(
            delta_times.unsqueeze(-1) * features_kern_linear +
            features_kern_bias,
            dim=1
        )  # shape = (bs, seq_len, out_channels)

        out += self.skip_connection(features)
        out = self.layer_norm(out)

        return out


class ContinuousConv1DSim(ContinuousConvolutionBase):
    """
    ContinuousConv1DSim extends ContinuousConvolutionBase to simulate additional time points between the real events.
    This layer is particularly useful in scenarios where the model needs to infer outputs at arbitrary time points
    between the observed events, based on the learned continuous convolution operation.

    Inherits from ContinuousConvolutionBase.
    """

    def __init__(
            self,
            kernel_size: int,
            input_channels: int,
            output_channels: int
    ):
        """
        Initializes the ContinuousConv1DSim layer, adjusting the kernel size to accommodate simulated time points.

        Parameters:
            kernel_size (int): The size of the convolution kernel, adjusted for simulated time points.
            input_channels (int): The number of input channels.
            output_channels (int): The number of output channels.
        """
        super().__init__(
            kernel_size=kernel_size + 1,
            input_channels=input_channels,
            output_channels=output_channels,
            dilation=1
        )
        self.kernel_network_weight = nn.Linear(input_channels, output_channels, bias=False)
        self.kernel_network_bias = nn.Parameter(
            torch.full(size=(input_channels, output_channels), fill_value=1 / output_channels))

    def forward(
            self,
            times: torch.Tensor,
            features: torch.Tensor,
            sample: torch.Tensor,
            scale: bool = True
    ) -> torch.Tensor:
        """
        Performs the forward pass of the ContinuousConv1DSim layer, incorporating simulated time points.

        Parameters:
            times (torch.Tensor): A tensor of shape (batch_size, seq_len) containing the times of each real event.
            features (torch.Tensor): A tensor of shape (batch_size, seq_len, input_channels) containing the features
                                     associated with each real event.
            sample (torch.Tensor): A tensor of shape (sim_size) containing uniformly sampled time points
                                            between real events, used for simulation.

        Returns:
            torch.Tensor: The output of the layer after applying the continuous convolution operation to both real
                          and simulated time points. The output tensor's shape accounts for the interpolated outputs
                          at simulated time points, typically larger than the input sequence length.
        """

        assert torch.all(sample == sample.sort().values)

        modified_features = self.kernel_network_weight(features)
        features_bias = features @ self.kernel_network_bias  # shape = (bs, seq_len, out_channels)

        all_features = torch.concat([modified_features, features_bias], dim=-1)

        delta_times, features_kern = self.construct_conv_matrix(
            times,
            all_features
        )
        features_kern_linear = features_kern[..., :self.output_channels]
        features_kern_bias = features_kern[..., self.output_channels:]

        delta_t_scale = delta_times[:, -2, 1:]  # equivalent to the times[1:] - times[:-1]

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
                    delta_t_scale[:, None, :, None, None] *
                    sample[None, None, None, :, None] * features_kern_linear[:, 1:, :-1, None, :] if scale
                    else sample[None, None, None, :, None] * features_kern_linear[:, 1:, :-1, None, :],
                    dim=1
                )
        )

        values_before_sim = real_values[:, :-1, :].unsqueeze(2)
        values_after_sim = real_values[:, -1:, :]

        bs, _, out_channels = values_after_sim.shape

        out = torch.cat([values_before_sim, sim_values], dim=2).reshape(bs, -1, out_channels)
        out = torch.cat([out, values_after_sim], dim=1)

        return out
