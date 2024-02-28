from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Predictions:
    loss: Optional[torch.Tensor] = None
    metrics: Optional[dict[str, float]] = None
