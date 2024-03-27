import numpy as np
import torch
import torch.nn as nn

from frozendict import frozendict

from src.models.components.cotic.head.structures import Predictions
from src.utils.data_utils.normalizers import Normalizer
from src.utils.metrics import LogCoshLoss

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression


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
            normalizer: Normalizer,
            stage: str
    ) -> Predictions:
        return_times = self.compute_return_times(embeddings.detach(), non_pad_mask)
        event_type_scores = self.compute_event_type_scores(embeddings.detach(), non_pad_mask)

        loss = self.time_loss_coeff * self.return_time_loss(
            return_times[:, :-1][non_pad_mask[:, 1:]],
            (times[:, 1:] - times[:, :-1] - 1)[non_pad_mask[:, 1:]]
        ) + self.type_loss_coeff * self.event_type_loss(
            event_type_scores[:, :-1, :].transpose(1, 2),
            (events - 1)[:, 1:]
        )

        metrics = dict()

        metrics['return_time_mae'] = torch.mean(
            torch.abs(
                normalizer.denormalize(
                    return_times[:, :-1][non_pad_mask[:, 1:]] + 1 - (times[:, 1:] - times[:, :-1])[non_pad_mask[:, 1:]]
                )
            )
        ).item()

        metrics['event_type_accuracy'] = torch.mean(
            (event_type_scores[:, :-1, :].argmax(dim=-1) == (events - 1)[:, 1:])[non_pad_mask[:, 1:]].double()
        ).item()

        return Predictions(loss=loss, metrics=metrics)


class DownstreamHeadSklearnLinear:
    def __init__(
        self,
        nb_filters: int,
        num_types: int,
        fit_every_n_epochs: int = 10
    ) -> None:
        self.return_time_model = LinearRegression()
        self.return_time_model.fit(
            np.random.random((100, nb_filters)), np.random.random(100)
        )
        self.event_type_model = LogisticRegression().fit(
            np.random.random((100, nb_filters)), np.random.randint(0, num_types, 100)
        )

        self.X = []
        self.y_times = []
        self.y_types = []

        self.epoch = 1
        self.change = False
        self.fit_every_n_epochs = fit_every_n_epochs

    def compute_return_times(
            self,
            embeddings: np.array,
            non_pad_mask: np.array,
    ) -> torch.Tensor:
        out = self.return_time_model.predict(embeddings[:, :-1][non_pad_mask[:, 1:]])

        return out

    def compute_event_type_scores(
            self,
            embeddings: np.array,
            non_pad_mask: np.array,
    ) -> torch.Tensor:
        out = self.event_type_model.predict(embeddings[:, :-1][non_pad_mask[:, 1:]])

        return out

    def __call__(
            self,
            times: torch.Tensor,
            events: torch.Tensor,
            embeddings: torch.Tensor,
            non_pad_mask: torch.Tensor,
            normalizer: Normalizer,
            stage: str
    ) -> Predictions:
        times = times.detach().cpu().numpy()
        events = events.detach().cpu().numpy()
        embeddings = embeddings.detach().cpu().numpy()
        non_pad_mask = non_pad_mask.detach().cpu().numpy()

        if stage == "train" and self.epoch % self.fit_every_n_epochs == 0:
            self.X.append(embeddings[:, :-1, :][non_pad_mask[:, 1:], :])
            self.y_times.append((times[:, 1:] - times[:, :-1] - 1)[non_pad_mask[:, 1:]])
            self.y_types.append((events[:, 1:] - 1)[non_pad_mask[:, 1:]])

        if stage != "train" and len(self.X) > 0:
            self.return_time_model.fit(
                np.concatenate(self.X), np.concatenate(self.y_times)
            )
            self.event_type_model.fit(
                np.concatenate(self.X), np.concatenate(self.y_types)
            )
            self.X = []
            self.y_times = []
            self.y_types = []

        if stage == "train":
            self.change = True
        if stage != "train" and self.change:
            self.change = False
            self.epoch += 1

        return_times = self.compute_return_times(embeddings, non_pad_mask)
        event_types = self.compute_event_type_scores(embeddings, non_pad_mask)

        metrics = dict()

        metrics['return_time_mae'] = np.mean(
            np.abs(
                normalizer.denormalize(
                    return_times + 1 - (times[:, 1:] - times[:, :-1])[non_pad_mask[:, 1:]]
                )
            )
        )

        metrics['event_type_accuracy'] = np.mean(
            (event_types == (events - 1)[:, 1:][non_pad_mask[:, 1:]]).astype(float)
        )

        return Predictions(metrics=metrics)


class ProbabilisticDownstreamHead:
    def __init__(self, compute_every_n_epochs: int = 10, sub_batch_size: int = 100) -> None:
        self.epoch = 0
        self.training = False
        self.batch_index = 0
        self.change = False
        self.compute_every_n_epochs = compute_every_n_epochs
        self.returns = dict(train=[], val=[])
        self.sub_batch_size = sub_batch_size

    @staticmethod
    def compute_cumulative_trapezoidal(times, values, interpolate_initial=True):
        if interpolate_initial:
            times = torch.concat([torch.zeros_like(times[:, 0:1]).to(times.device), times], dim=1)
            values = torch.concat([torch.ones(values[:, 0:1].shape).to(times.device) * values[:, 0:1], values], dim=1)

        # Calculate the width of each trapezoid (time intervals)
        delta_times = times[:, 1:] - times[:, :-1]

        # Calculate the average height of each trapezoid (average intensity)
        avg_intensity = (values[:, 1:] + values[:, :-1]) / 2

        # Calculate the area of each trapezoid
        trapezoid_areas = delta_times * avg_intensity

        # Initialize the cumulative intensity tensor with zeros
        cumulative_intensity = torch.zeros_like(values)

        # Compute the cumulative sum of trapezoid areas for cumulative intensity
        cumulative_intensity[:, 1:] = torch.cumsum(trapezoid_areas, dim=1)

        return times, cumulative_intensity

    def compute_return_times(
            self,
            delta_times: torch.Tensor,
            intensity: torch.Tensor,
            non_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        intensity = intensity.sum(-1)

        bs, seq_len, sim_size = intensity.shape
        intensity = intensity.reshape(-1, sim_size)
        delta_times = delta_times.unsqueeze(0)

        times, cumulative_intensity = self.compute_cumulative_trapezoidal(delta_times, intensity)
        _, out = self.compute_cumulative_trapezoidal(times, torch.exp(-cumulative_intensity), False)

        out = out[:,-1].reshape(bs, seq_len)

        out[~non_pad_mask[:, 1:]] = 0

        return out

    @staticmethod
    def interpolate_intensity(return_times, times, intensity):
        bs, seq_len, sim_size, num_types = intensity.shape

        # Expand dimensions for broadcasting
        times_expanded = times.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # Shape: (1, 1, sim_size, 1)
        return_times_expanded = return_times.unsqueeze(-1).unsqueeze(-1)  # Shape: (bs, seq_len, 1, 1)

        # Compute absolute differences
        diffs = torch.abs(return_times_expanded - times_expanded)

        # Find indices of the closest smaller or equal and the closest larger times for each return_time
        _, indices_left = torch.min(
            (diffs + torch.where(return_times_expanded < times_expanded, torch.tensor(float('inf')), 0)), dim=2)
        _, indices_right = torch.min(
            (diffs + torch.where(return_times_expanded > times_expanded, torch.tensor(float('inf')), 0)), dim=2)

        # Gather the intensity values at these indices
        intensity_left = torch.gather(intensity, 2, indices_left.unsqueeze(-1).expand(-1, -1, -1, intensity.size(-1)))
        intensity_right = torch.gather(intensity, 2, indices_right.unsqueeze(-1).expand(-1, -1, -1, intensity.size(-1)))

        # Compute the weights for interpolation
        times_left = torch.gather(times_expanded.squeeze(-1).repeat(bs, seq_len, 1), 2, indices_left)
        times_right = torch.gather(times_expanded.squeeze(-1).repeat(bs, seq_len, 1), 2, indices_right)
        weights_left = (times_right - return_times_expanded.squeeze(-1)) / (
                    times_right - times_left + 1e-8)  # Add epsilon to avoid division by zero
        weights_right = (return_times_expanded.squeeze(-1) - times_left) / (times_right - times_left + 1e-8)

        # Perform interpolation
        intensity_in_point = weights_left.unsqueeze(-1) * intensity_left + weights_right.unsqueeze(-1) * intensity_right

        # Remove the extra dimension added by unsqueeze
        intensity_in_point = intensity_in_point.squeeze(2)  # Shape: (bs, seq_len, num_types)

        return intensity_in_point

    def compute_event_type_scores(
            self,
            return_times: torch.Tensor,
            delta_times: torch.Tensor,
            intensity: torch.Tensor,
            non_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = self.interpolate_intensity(return_times, delta_times, intensity)

        scores[~non_pad_mask[:, 1:]] = 0

        return torch.argmax(scores, dim=-1)

    def compute_prediction(
            self,
            times: torch.Tensor,
            events: torch.Tensor,
            embeddings: torch.Tensor,
            non_pad_mask: torch.Tensor,
            normalizer: Normalizer,
            stage: str,
            intensity_head: nn.Module
    ) -> Predictions:
        embeddings = embeddings.detach()
        delta_times = torch.arange(1e-8, 13, 1e-2).to(times.device)

        with torch.no_grad():
            intensity = []
            for i in range(0, len(delta_times), self.sub_batch_size):
                sub_batch_size = len(delta_times[i:i + self.sub_batch_size])
                sub_intensity = intensity_head.compute_lambdas(
                    times,
                    embeddings,
                    delta_times[i:i + self.sub_batch_size],
                    scale=False
                )
                mask = torch.ones(sub_intensity.shape[1]).bool().to(times.device)
                mask[::(sub_batch_size + 1)] = False
                sub_intensity = sub_intensity[:, mask, :].reshape(times.shape[0], -1, sub_batch_size, sub_intensity.shape[-1])
                intensity.append(sub_intensity)

            intensity = torch.cat(intensity, dim=2)

        return_times = self.compute_return_times(delta_times, intensity, non_pad_mask)
        event_types = self.compute_event_type_scores(return_times, delta_times, intensity, non_pad_mask)

        metrics = dict()

        metrics['return_time_mae'] = torch.abs(
            normalizer.denormalize(
                return_times[non_pad_mask[:, 1:]] - (times[:, 1:] - times[:, :-1])[non_pad_mask[:, 1:]]
            ).cpu()
        )

        metrics['event_type_accuracy'] = (event_types == (events - 1)[:, 1:])[non_pad_mask[:, 1:]].cpu().float()

        return Predictions(metrics=metrics)

    def __call__(
        self,
        times: torch.Tensor,
        events: torch.Tensor,
        embeddings: torch.Tensor,
        non_pad_mask: torch.Tensor,
        normalizer: Normalizer,
        stage: str,
        intensity_head: nn.Module
    ) -> Predictions:
        if stage == 'test':
            return self.compute_prediction(
                times,
                events,
                embeddings,
                non_pad_mask,
                normalizer,
                stage,
                intensity_head
            )

        if stage != "train" and self.training:
            self.change = True
        if stage == "train":
            self.training = True
        if stage == "train" and self.change:
            if (self.epoch + 1) % self.compute_every_n_epochs == 0:
                self.returns['train'] = []
                self.returns['val'] = []
            self.change = False
            self.epoch += 1

        if self.epoch % self.compute_every_n_epochs == 0:
            self.returns[stage].append(self.compute_prediction(
                times,
                events,
                embeddings,
                non_pad_mask,
                normalizer,
                stage,
                intensity_head
            ))
            return self.returns[stage][-1]

        self.batch_index += 1

        return self.returns[stage][self.batch_index % len(self.returns[stage])]
