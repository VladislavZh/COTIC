import torch.nn as nn
import torch


class LinearKernel(nn.Module):
    """
    A PyTorch Module that implements a linear kernel. This module takes an input tensor of shape (*, 1)
    and returns a tensor of shape (*, in_channels, out_channels). The module applies a linear transformation
    followed by a dropout and reshaping operation.

    Attributes:
    - in_channels (int): The number of input channels.
    - out_channels (int): The number of output channels.
    - dropout (float): The dropout rate used in the dropout layer.

    Methods:
    - recreate: Creates a new instance of LinearKernel with modified in_channels.
    - compute_addition: Computes and returns the matrix multiplication of input with the layer's weights.
    - forward: Defines the computation performed at every call, applying linear transformation and reshaping the output.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0) -> None:
        """
        Initializes the LinearKernel module.

        Args:
        - in_channels (int): The number of input channels.
        - out_channels (int): The number of output channels.
        - dropout (float): The dropout rate, used for regularization. Default: 0.
        """
        super().__init__()
        self.args = [in_channels, out_channels, dropout]

        self.layer = nn.Linear(1, in_channels * out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = nn.Dropout(dropout)
        self.__dropout_value = dropout

    def validate_addition_computation(self) -> None:
        """
        Validates that the LinearKernel's compute_addition functionality can be used.

        This method checks if the dropout value is set to zero. The compute_addition functionality
        is only applicable when there is no dropout (dropout = 0). If the dropout value is not zero,
        this method raises a RuntimeError indicating that compute_addition cannot be used.

        Raises:
        - RuntimeError: If the dropout value is not zero, indicating that compute_addition is not applicable.
        """
        if self.__dropout_value != 0:
            raise RuntimeError(f"LinearKernel is expected to be used with the compute_addition functionality, "
                               f"it is applicable with dropout = 0 only, yet it is {self.__dropout_value}")

    def recreate(self, in_channels: int) -> 'LinearKernel':
        """
        Creates a new instance of LinearKernel with a different number of input channels.

        Args:
        - in_channels (int): The number of input channels for the new instance.

        Returns:
        - LinearKernel: A new instance of LinearKernel with the specified in_channels.
        """
        args = self.args.copy()
        args[0] = in_channels
        return type(self)(*args)

    def compute_addition(self, delta_x: torch.Tensor) -> torch.Tensor:
        """
        Computes the matrix multiplication of input delta_x with the layer's weights.

        Args:
        - delta_x (torch.Tensor): The input tensor to be multiplied with the layer's weights.

        Returns:
        - torch.Tensor: The result of the matrix multiplication.
        """
        shape = list(delta_x.shape)[:-1]
        shape += [self.in_channels, self.out_channels]
        return (delta_x @ self.layer.weight.T).reshape(*shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the LinearKernel module.

        Applies a linear transformation followed by dropout and reshaping to the input tensor.

        Args:
        - x (torch.Tensor): The input tensor of shape (*, 1).

        Returns:
        - torch.Tensor: The output tensor of shape (*, in_channels, out_channels).
        """
        shape = list(x.shape)[:-1]
        shape += [self.in_channels, self.out_channels]
        x = self.dropout(self.layer(x))
        x = x.reshape(*shape)

        return x
