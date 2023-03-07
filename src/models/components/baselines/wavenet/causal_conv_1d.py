import torch
import torch.nn as nn
import torch.nn.init as init


class DilatedCausalConv1d(nn.Module):
    """Class for 1-dimensional dilated causal convolution."""

    def __init__(
        self,
        hyperparams: dict,
        dilation_factor: int,
        in_channels: int,
        causal: bool = True,
    ) -> None:
        """Initialize 1-dimensional dilated causal convolution.

        args:
            hyperparams - dictionary with hyper-parameters (from config)
            dilation_factor - dilation (spacing between the kernel points)
            in_channels - number of input channels
            causal - if True, make causal convolution
        """
        super().__init__()

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight.data)
                init.zeros_(m.bias.data)

        self.causal = causal

        # no future time stamps available
        if self.causal:
            self.padding = (hyperparams["kernel_size"] - 1) * dilation_factor
        else:
            self.padding = ((hyperparams["kernel_size"] - 1) * dilation_factor) // 2

        self.dilation_factor = dilation_factor
        self.dilated_causal_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hyperparams["nb_filters"],
            kernel_size=hyperparams["kernel_size"],
            dilation=dilation_factor,
            padding=self.padding,
        )
        self.dilated_causal_conv.apply(weights_init)

        self.skip_connection = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hyperparams["nb_filters"],
            kernel_size=1,
        )
        self.skip_connection.apply(weights_init)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def _make_causal(self, x: torch.Tensor) -> torch.Tensor:
        """Make causal convolution, if specified."""
        return x if not self.causal else x[..., : -self.padding]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x1 = self.leaky_relu(self._make_causal(self.dilated_causal_conv(x)))
        x2 = self.skip_connection(x) + x1
        return x2
