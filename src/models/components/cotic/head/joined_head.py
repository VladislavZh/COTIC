import torch
from torch import nn

from src.models.components.cotic.head.structures import Predictions
from src.utils.data_utils.normalizers import Normalizer


class JoinedHead(nn.Module):
    """
    Class for the joined head of the model.
    """

    def __init__(
            self,
            intensity_head: nn.Module,
            downstream_head: nn.Module,
            uniform_sample_size: int
    ) -> None:
        super().__init__()
        self.intensity_head = intensity_head
        self.downstream_head = downstream_head
        self.uniform_sample_size = uniform_sample_size

    def forward(
        self,
        times: torch.Tensor,
        events: torch.Tensor,
        embeddings: torch.Tensor,
        non_pad_mask: torch.Tensor,
        normalizer: Normalizer,
        stage: str
    ) -> tuple[Predictions, ...]:
        uniform_sample = torch.rand(self.uniform_sample_size).to(times.device).sort().values
        intensity_predictions = self.intensity_head(
            times,
            events,
            embeddings,
            non_pad_mask,
            uniform_sample
        )
        downstream_predictions = self.downstream_head(
            times,
            events,
            embeddings,
            non_pad_mask,
            normalizer,
            stage
        )

        return intensity_predictions, downstream_predictions


class ProbabilisticJoinedHead(JoinedHead):
    def forward(
        self,
        times: torch.Tensor,
        events: torch.Tensor,
        embeddings: torch.Tensor,
        non_pad_mask: torch.Tensor,
        normalizer: Normalizer,
        stage: str
    ) -> tuple[Predictions, ...]:
        uniform_sample = torch.rand(self.uniform_sample_size).to(times.device).sort().values
        intensity_predictions = self.intensity_head(
            times,
            events,
            embeddings,
            non_pad_mask,
            uniform_sample
        )
        downstream_predictions = self.downstream_head(
            times,
            events,
            embeddings,
            non_pad_mask,
            normalizer,
            stage,
            self.intensity_head
        )

        return intensity_predictions, downstream_predictions
