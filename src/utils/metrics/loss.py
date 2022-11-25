import torch
import math

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def __log_cosh(x: torch.Tensor) -> torch.Tensor:
        out = x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
        out[torch.abs(x) > 20] = torch.abs(x[torch.abs(x) > 20])
        return out

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return torch.mean(self.__log_cosh(y_pred - y_true))
