import torch
import math

def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

    return torch.mean(_log_cosh(y_pred - y_true))

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __log_cosh(x: torch.Tensor) -> torch.Tensor:
        out = x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
        out[x > 20] = x
        return out

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return torch.mean(self.__log_cosh(y_pred - y_true))
