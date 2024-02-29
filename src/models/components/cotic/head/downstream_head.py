import torch
import torch.nn as nn

from frozendict import frozendict

from src.models.components.cotic.head.structures import Predictions
from src.utils.data_utils.normalizers import Normalizer
from src.utils.metrics import LogCoshLoss


class DownstreamHead(nn.Module):
    """
    A downstream head module for a neural network model, designed for tasks involving return time prediction and event type classification.

    Attributes:
        activation: The activation function used in the model, set to LeakyReLU with a negative slope of 0.1.
        return_time_layers: A sequence of Linear layers for predicting return times of events.
        event_type_layers: A sequence of Linear layers for classifying event types.
        return_time_loss: The loss function used for return time predictions, set to LogCoshLoss.
        event_type_loss: The loss function used for event type classification, set to CrossEntropyLoss.
        type_loss_coeff: A coefficient for scaling the event type loss component.
        time_loss_coeff: A coefficient for scaling the return time loss component.

    Parameters:
        nb_filters (int): The number of filters in the input feature dimension.
        mlp_layers (list[int]): A list specifying the size of each layer in the MLP.
        num_types (int): The number of unique event types for classification.
        type_loss_coeff (float): The coefficient for the event type loss term.
        time_loss_coeff (float): The coefficient for the return time loss term.
        reductions (dict): A dictionary specifying the reduction methods for each loss term.
    """
    def __init__(
        self,
        nb_filters: int,
        mlp_layers: list[int],
        num_types: int,
        type_loss_coeff: float = 1,
        time_loss_coeff: float = 1,
        reductions: dict = frozendict(type="mean", time="mean")
    ) -> None:
        super().__init__()
        self.activation = nn.LeakyReLU(0.1)
        self.return_time_layers = nn.ModuleList(
            [
                nn.Linear(in_channels, out_channels)
                for in_channels, out_channels in
                zip([nb_filters] + mlp_layers, mlp_layers + [1])
            ]
        )
        self.event_type_layers = nn.ModuleList(
            [
                nn.Linear(in_channels, out_channels)
                for in_channels, out_channels in
                zip([nb_filters] + mlp_layers, mlp_layers + [num_types])
            ]
        )

        self.return_time_loss = LogCoshLoss(reduction=reductions["time"])
        self.event_type_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction=reductions["type"])

        self.type_loss_coeff = type_loss_coeff
        self.time_loss_coeff = time_loss_coeff

    def compute_return_times(
            self,
            embeddings: torch.Tensor,
            non_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the return times for events based on their embeddings.

        Parameters:
            embeddings (torch.Tensor): The embeddings of the events.
            non_pad_mask (torch.Tensor): A mask indicating the non-padded elements in the sequence.

        Returns:
            torch.Tensor: The predicted return times for the events.
        """
        out = embeddings

        for layer in self.return_time_layers:
            out = self.activation(
                layer(out)
            )

        out[~non_pad_mask] = 0

        return out

    def compute_event_type_scores(
            self,
            embeddings: torch.Tensor,
            non_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the scores for event types based on their embeddings.

        Parameters:
            embeddings (torch.Tensor): The embeddings of the events.
            non_pad_mask (torch.Tensor): A mask indicating the non-padded elements in the sequence.

        Returns:
            torch.Tensor: The scores for each event type.
        """
        out = embeddings

        for layer in self.event_type_layers:
            out = self.activation(
                layer(out)
            )

        out[~non_pad_mask, :] = 0

        return out

    def forward(
            self,
            times: torch.Tensor,
            events: torch.Tensor,
            embeddings: torch.Tensor,
            non_pad_mask: torch.Tensor,
            normalizer: Normalizer
    ) -> Predictions:
        """
        Executes the forward pass of the DownstreamHead module, calculating the loss and performance metrics for a given batch of data.
        This method processes the event embeddings through separate layers to predict return times and event types, applies
        losses to these predictions, and computes metrics for model evaluation.

        Detaches the embeddings before passing them through the return time and event type layers to avoid backpropagating
        through the upstream components. Transposes the event type scores for compatibility with the loss function.

        Parameters:
            times (torch.Tensor): A tensor of shape [batch_size, sequence_length] containing the times at which events occur.
            events (torch.Tensor): A tensor of shape [batch_size, sequence_length] containing the indices of event types.
            embeddings (torch.Tensor): A tensor of shape [batch_size, sequence_length, embedding_dim] containing the embeddings for each event.
            non_pad_mask (torch.Tensor): A boolean tensor of shape [batch_size, sequence_length] indicating non-padded elements in the sequence.
            normalizer (Normalizer): An instance of a Normalizer class used to denormalize the predicted return times for metric calculation.

        Returns:
            Predictions: An object containing:
                - loss: The combined loss from the return time and event type predictions, scaled by their respective coefficients.
                - metrics: A dictionary containing 'return_time_mae' for the mean absolute error of return times and 'event_type_accuracy' for the accuracy of event type predictions.
        """
        return_times = self.compute_return_times(embeddings.detach(), non_pad_mask)
        event_type_scores = self.compute_event_type_scores(embeddings.detach(), non_pad_mask)

        loss = self.time_loss_coeff * self.return_time_loss(
            return_times[:, :-1][non_pad_mask[:, 1:]],
            (times[:, 1:] - times[:, :-1])[non_pad_mask[:, 1:]]
        ) + self.type_loss_coeff * self.event_type_loss(
            event_type_scores[:, :-1, :].transpose(1, 2),
            (events - 1)[:, 1:]
        )

        metrics = dict()

        metrics['return_time_mae'] = torch.mean(
            torch.abs(
                normalizer.denormalize(
                    return_times[:, :-1][non_pad_mask[:, 1:]] - (times[:, 1:] - times[:, :-1])[non_pad_mask[:, 1:]]
                )
            )
        ).item()

        metrics['event_type_accuracy'] = torch.mean(
            (event_type_scores[:, :-1, :].argmax(dim=-1) == (events - 1)[:, 1:])[non_pad_mask[:, 1:]].double()
        ).item()

        return Predictions(loss=loss, metrics=metrics)


class DownstreamHeadLinear(nn.Module):
    def __init__(
        self,
        nb_filters: int,
        num_types: int,
        type_loss_coeff: float = 1,
        time_loss_coeff: float = 1,
        reductions: dict = frozendict(type="mean", time="mean")
    ) -> None:
        super().__init__()
        self.return_time_layer = nn.Linear(nb_filters, 1)
        self.event_type_layer = nn.Linear(nb_filters, num_types)

        self.return_time_loss = LogCoshLoss(reduction=reductions["time"])
        self.event_type_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction=reductions["type"])

        self.type_loss_coeff = type_loss_coeff
        self.time_loss_coeff = time_loss_coeff

    def compute_return_times(
            self,
            embeddings: torch.Tensor,
            non_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.return_time_layer(embeddings)

        out[~non_pad_mask] = 0

        return out

    def compute_event_type_scores(
            self,
            embeddings: torch.Tensor,
            non_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.event_type_layer(embeddings)

        out[~non_pad_mask, :] = 0

        return out

    def forward(
            self,
            times: torch.Tensor,
            events: torch.Tensor,
            embeddings: torch.Tensor,
            non_pad_mask: torch.Tensor,
            normalizer: Normalizer
    ) -> Predictions:
        return_times = self.compute_return_times(embeddings.detach(), non_pad_mask)
        event_type_scores = self.compute_event_type_scores(embeddings.detach(), non_pad_mask)

        loss = self.time_loss_coeff * self.return_time_loss(
            return_times[:, :-1][non_pad_mask[:, 1:]],
            (times[:, 1:] - times[:, :-1])[non_pad_mask[:, 1:]]
        ) + self.type_loss_coeff * self.event_type_loss(
            event_type_scores[:, :-1, :].transpose(1, 2),
            (events - 1)[:, 1:]
        )

        metrics = dict()

        metrics['return_time_mae'] = torch.mean(
            torch.abs(
                normalizer.denormalize(
                    return_times[:, :-1][non_pad_mask[:, 1:]] - (times[:, 1:] - times[:, :-1])[non_pad_mask[:, 1:]]
                )
            )
        ).item()

        metrics['event_type_accuracy'] = torch.mean(
            (event_type_scores[:, :-1, :].argmax(dim=-1) == (events - 1)[:, 1:])[non_pad_mask[:, 1:]].double()
        ).item()

        return Predictions(loss=loss, metrics=metrics)