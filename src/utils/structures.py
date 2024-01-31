from dataclasses import dataclass
from typing import Optional

import torch


class Predictions:
    loss: Optional[torch.Tensor] = None


@dataclass
class DownstreamPredictions(Predictions):
    metrics: dict[str, float]
    loss: Optional[torch.Tensor] = None
