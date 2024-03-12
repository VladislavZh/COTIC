import torch
import torch.nn as nn

from src.models.components.cotic.cotic_layers import ContinuousConv1DSim
from src.models.components.cotic.head.structures import Predictions


class IntensityHead(nn.Module):
    """
    A neural network module for predicting event intensity functions in temporal point processes.

    This module is designed to compute the intensity function of events over time, given their embeddings.
    It utilizes a 1D convolution layer followed by a series of linear layers to process the embeddings and
    outputs the intensity of events at given times. The final output is processed through a Softplus activation
    function to ensure non-negativity.

    Parameters:
    - kernel_size (int): The kernel size for the convolution layer.
    - nb_filters (int): The number of filters (and output channels) for the convolution layer.
    - mlp_layers (list[int]): A list specifying the size of each linear layer in the multi-layer perceptron (MLP).
    - num_types (int): The number of different event types the model is expected to handle.

    Attributes:
    - convolution (ContinuousConv1DSim): A 1D convolution layer customized for continuous input data.
    - activation (nn.LeakyReLU): A leaky ReLU activation function applied after convolution and each linear layer.
    - layers (nn.ModuleList): A list of linear layers constituting the MLP part of the network.
    - softplus (nn.Softplus): A Softplus activation function applied to the output to ensure non-negative intensities.

    Methods:
    - compute_lambdas(times, embeddings, non_pad_mask, uniform_sample): Computes the intensity function λ(t) for given
      times and embeddings.
    - forward(times, events, embeddings, non_pad_mask, uniform_sample): Processes the input data through the network
      and computes the loss based on the predicted intensities and actual events.

    The forward pass of this module computes the loss function for a given batch of event sequences, which can be used
    for training the model in the context of temporal point processes.
    """
    def __init__(
        self,
        kernel_size: int,
        nb_filters: int,
        mlp_layers: list[int],
        num_types: int
    ) -> None:
        super().__init__()
        self.convolution = ContinuousConv1DSim(
            kernel_size, nb_filters, nb_filters
        )
        self.activation = nn.LeakyReLU(0.1)
        self.layers = nn.ModuleList(
            [
                nn.Linear(in_channels, out_channels)
                for in_channels, out_channels in
                zip([nb_filters] + mlp_layers, mlp_layers + [num_types])
            ]
        )

        self.softplus = nn.Softplus(100)
        self.num_types = num_types

    def compute_lambdas(
            self,
            times: torch.Tensor,
            embeddings: torch.Tensor,
            uniform_sample: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the intensity function λ(t) for given times and embeddings.

        This method applies a convolution over the embeddings with respect to the specified times, followed by
        a series of linear transformations. Each step involves an activation function to introduce non-linearities.
        The final output is passed through a Softplus activation to ensure the intensities are non-negative.

        Parameters:
        - times (torch.Tensor): A tensor of shape [batch_size, sequence_length] containing the times of events.
        - embeddings (torch.Tensor): A tensor of shape [batch_size, sequence_length, embedding_dim] containing
          the embeddings of events.
        - non_pad_mask (torch.Tensor): A boolean tensor of shape [batch_size, sequence_length] indicating non-padded
          elements in the sequence.
        - uniform_sample (torch.Tensor): A tensor containing uniformly sampled time points used for intensity
          integration.

        Returns:
        - torch.Tensor: A tensor of shape [batch_size, sequence_length, num_types] representing the intensity
          of each event type at each time point in the sequence.
        """
        continuous_sample_embeddings = self.activation(
            self.convolution(
                times,
                embeddings,
                uniform_sample
            )
        )

        for layer in self.layers:
            continuous_sample_embeddings = self.activation(
                layer(continuous_sample_embeddings)
            )

        return self.softplus(continuous_sample_embeddings)

    def forward(
            self,
            times: torch.Tensor,
            events: torch.Tensor,
            embeddings: torch.Tensor,
            non_pad_mask: torch.Tensor,
            uniform_sample: torch.Tensor
    ) -> Predictions:
        """
        Forward pass of the model, calculating the loss for a given batch of data. The loss is computed by combining
        the negative log-likelihood of observed events with the integral of the intensity function over the observation
        window. The intensity functions are determined for the provided times and embeddings, and the loss is adjusted
        for sequences that are not padded.

        The method involves the following steps:
        1. Computing the intensity functions for each time point in 'times' using the provided 'embeddings'.
        2. Selecting the intensities corresponding to the actual 'events'.
        3. Calculating the negative log-likelihood for the observed 'events'.
        4. Integrating the intensity function over the observation window using 'uniform_sample' for numerical integration.
        5. Adjusting the loss calculation for non-padded elements using 'non_pad_mask'.

        Parameters:
        - times (torch.Tensor): Tensor of shape [batch_size, sequence_length], with times of events.
        - events (torch.Tensor): Tensor of shape [batch_size, sequence_length], with indices of event types.
        - embeddings (torch.Tensor): Tensor of shape [batch_size, sequence_length, embedding_dim], with embeddings of events.
        - non_pad_mask (torch.Tensor): Boolean tensor of shape [batch_size, sequence_length], indicating non-padded elements.
        - uniform_sample (torch.Tensor): Tensor with uniformly sampled time points for intensity integration.

        Returns:
        - Predictions: An object containing the computed loss for the batch. The loss combines the negative log likelihood
          of the observed events with the integral of the intensity function, adjusted for non-padding elements.
        """
        lambdas = self.compute_lambdas(
            times,
            embeddings,
            uniform_sample
        )

        events_index = (events - 1)
        events_index[events_index == -1] = 0

        events_lambdas = lambdas[:, ::(len(uniform_sample) + 1)].gather(2, events_index.unsqueeze(-1)).squeeze()

        mask = torch.ones(lambdas.shape[1]).bool().to(times.device)
        mask[::(len(uniform_sample) + 1)] = False
        non_event_lambdas = lambdas[:, mask, :]
        integral = non_event_lambdas.reshape(times.shape[0], -1, len(uniform_sample), self.num_types).sum((-2, -1)) * (times[:, 1:] - times[:, :-1]) / len(
            uniform_sample)

        loss = torch.sum(-torch.log(events_lambdas + 1e-8) * non_pad_mask) + torch.sum(integral * non_pad_mask[:, 1:])
        loss /= torch.sum(non_pad_mask)

        return Predictions(loss=loss)


class IntensityHeadLinear(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        nb_filters: int,
        num_types: int
    ) -> None:
        super().__init__()
        self.convolution = ContinuousConv1DSim(
            kernel_size, nb_filters, nb_filters
        )
        self.activation = nn.LeakyReLU(0.1)
        self.layer = nn.Linear(nb_filters, num_types)

        self.softplus = nn.Softplus(num_types)
        self.num_types = num_types

    def compute_lambdas(
            self,
            times: torch.Tensor,
            embeddings: torch.Tensor,
            uniform_sample: torch.Tensor,
            scale: bool = True
    ) -> torch.Tensor:
        continuous_sample_embeddings = self.activation(
            self.convolution(
                times,
                embeddings,
                uniform_sample,
                scale
            )
        )

        continuous_sample_embeddings = self.layer(continuous_sample_embeddings)

        return self.softplus(continuous_sample_embeddings)

    def forward(
            self,
            times: torch.Tensor,
            events: torch.Tensor,
            embeddings: torch.Tensor,
            non_pad_mask: torch.Tensor,
            uniform_sample: torch.Tensor
    ) -> Predictions:
        lambdas = self.compute_lambdas(
            times,
            embeddings,
            uniform_sample
        )

        events_index = (events - 1)
        events_index[events_index == -1] = 0

        events_lambdas = lambdas[:, ::(len(uniform_sample) + 1)].gather(2, events_index.unsqueeze(-1)).squeeze()

        mask = torch.ones(lambdas.shape[1]).bool().to(times.device)
        mask[::(len(uniform_sample) + 1)] = False
        non_event_lambdas = lambdas[:, mask, :]
        integral = non_event_lambdas.reshape(times.shape[0], -1, len(uniform_sample), self.num_types).sum((-2, -1)) * (times[:, 1:] - times[:, :-1]) / len(
            uniform_sample)

        loss = torch.sum(-torch.log(events_lambdas + 1e-8) * non_pad_mask) + torch.sum(integral * non_pad_mask[:, 1:])
        loss /= torch.sum(non_pad_mask)

        return Predictions(loss=loss)
