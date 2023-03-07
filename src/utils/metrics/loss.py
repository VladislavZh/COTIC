import torch
import math


class LogCoshLoss(torch.nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        """Initialize LogCoshLoss class."""
        super().__init__()
        if reduction not in ["mean", "sum"]:
            raise ValueError("LogCoshLoss: reduction not in ['mean', 'sum']")
        self.__reduction = reduction

    @staticmethod
    def __log_cosh(x: torch.Tensor) -> torch.Tensor:
        """Compute LogCosh function.

        :param x: input

        :return: out = LogCosh(x), use absolute value if abs(out) > 10
        """
        out = x + torch.nn.functional.softplus(-2.0 * x) - math.log(2.0)
        out[torch.abs(x) > 10] = torch.abs(x[torch.abs(x) > 10])
        return out

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Compute LogCosh loss function.

        :param y_true: torch.Tensor with true event times
        :param p_pred: torch.Tensor with predicted event types

        :return: LogCosh(y_true - y_pred) averaged / summed over a batch
        """
        if self.__reduction == "mean":
            return torch.mean(self.__log_cosh(y_pred - y_true))
        else:
            return torch.sum(self.__log_cosh(y_pred - y_true))
