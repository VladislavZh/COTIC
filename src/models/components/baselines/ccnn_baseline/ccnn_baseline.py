import torch
import torch.nn as nn

from typing import Tuple

from .cont_cnn_layers import ContConv1d


class BaselineCCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        nb_filters: int,
        nb_layers: int,
        num_types: int,
        nb_layers_discr: int,
        discr_conv_out_channels: int,
        padding_discr: int,
        kernel: nn.Module,
        pp_output_layer: nn.Module,
    ) -> None:
        """Initialize baseline CCNN model.

        args:
            in_channels - number of input channels for 1d continuous convolutions
            kernel_size - kernel size for 1d continuous convolutions
            nb_filters - number of filters for 1d continuous convolutions
            nb_layers - number of continuous convolutional layers
            num_types - number of event types in the dataset
            nb_layers_discr - number of discrete convolutional layers
            discr_conv_out_channels - out dimensionality of distrete convolutions
            padding_discr - padding (for discrete convolutions)
            kernel - kernel model (for continuous convolutions)
            pp_output_layer - output layers producing log-likelohood function
        """

        super().__init__()
        self.event_emb = nn.Embedding(num_types + 2, in_channels, padding_idx=0)

        self.in_channels = [in_channels] + [nb_filters] * nb_layers
        include_zero_lag = [True] + [True] * nb_layers
        self.dilation_factors = [2**i for i in range(0, nb_layers)]
        self.dilation_factors_discr = [2**j for j in range(0, nb_layers_discr)]

        self.num_types = num_types
        self.nb_layers = nb_layers
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.nb_layers_discr = nb_layers_discr
        self.discr_conv_out_channels = discr_conv_out_channels
        self.padding_discr = padding_discr

        self.cont_convs = nn.ModuleList(
            [
                ContConv1d(
                    kernel.recreate(self.in_channels[i]),
                    self.kernel_size,
                    self.in_channels[i],
                    self.nb_filters,
                    self.dilation_factors[i],
                    include_zero_lag[i],
                )
                for i in range(self.nb_layers)
            ]
        )

        self.pp_output_layer = pp_output_layer

        self.discr_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.nb_filters,
                    out_channels=self.discr_conv_out_channels,
                    kernel_size=self.kernel_size,
                    padding=self.padding_discr,
                    dilation=self.dilation_factors_discr[j],
                )
                for j in range(self.nb_layers_discr)
            ]
        )

    def add_bos(
        self,
        event_times: torch.Tensor,
        event_types: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add zero time at the begining of all the sequences, increase sequence lengths.

        args:
            event_times - batch of sequences with event times
            event_types - batch of sequences with event types
            lengths - sequence lengths

        returns:
           event_times - event times prepended with 0
           event_types - event types, add special type for zero events
           lengths - sequence lengths incremented by 1

        """
        bs = event_times.shape[0]
        event_times = torch.concat(
            [torch.zeros(bs, 1).to(event_times.device), event_times], dim=1
        )
        max_event_type = torch.max(event_types) + 1
        tmp = (torch.ones(bs, 1).to(event_types.device) * max_event_type).long()
        event_types = torch.concat([tmp, event_types], dim=1)
        lengths += 1
        return event_times, event_types, lengths

    def differentiate(self, in_times: torch.Tensor) -> torch.Tensor:
        """Differentiate sequence with event times: compute time increments.

        args:
            in_times - sequence of event times to be differentiated

        returns:
            dt - sequence of time increments
        """
        diff = torch.diff(in_times)
        # add 0 as the first element - no time from the previous event
        zeros = torch.zeros(in_times.shape[0]).unsqueeze(dim=1).to(in_times.device)
        dt = torch.hstack((zeros, diff))
        return dt

    def get_enc_output(
        self, event_times: torch.Tensor, event_types: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the convolutional part of the model.
        Return encoder output - hidden state to be fed into pp_output_layer.

        args:
            event_times - torch.Tensor, shape = (bs, L) event times
            event_types - torch.Tensor, shape = (bs, L) event types

        returns:
            encoded_output - hidden state to be fed into pp_output_layer
        """
        lengths = torch.sum(event_types.ne(0).type(torch.float), dim=1).long()
        event_times, event_types, lengths = self.add_bos(
            event_times, event_types, lengths
        )

        non_pad_mask = event_types.ne(0)

        enc_output = self.event_emb(event_types)

        for cont_conv in self.cont_convs:
            enc_output = torch.nn.functional.leaky_relu(
                cont_conv(event_times, enc_output, non_pad_mask), 0.1
            )

        enc_output = enc_output.transpose(1, 2)

        for discr_conv in self.discr_convs:
            enc_output = torch.nn.functional.leaky_relu(discr_conv(enc_output), 0.1)

        enc_output = enc_output.transpose(1, 2)

        return enc_output

    def forward(
        self, event_times: torch.Tensor, event_types: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the model.

        args:
            event_times - batch of sequences with event times
            event_types - batch of sequences with event types

        returns out - a tuple of:
            multinomial - predicted event type scores, shape is (batch_size, seq_len, num_types)
            log_likelihood - value of log-likelihoos function, shape is (batch_size, seq_len, 1)
        """

        # TODO: fix this
        _lengths = torch.sum(event_types.ne(0).type(torch.float), dim=1).long()
        _event_times, _, _ = self.add_bos(event_times, event_types, _lengths)

        dt = self.differentiate(_event_times)

        out = self.get_enc_output(event_times, event_types)

        out = self.pp_output_layer(out, dt)
        return out

    def get_lambda_0(
        self, event_times: torch.Tensor, event_types: torch.Tensor
    ) -> torch.Tensor:
        """Get lambda_0 (aka intensities) for return time estimation through integration.

        args:
            event_times - batch of sequences with event times
            event_types - batch of sequences with event types

        returns:
            lambda_0 - intensities
        """
        enc_output = self.get_enc_output(event_times, event_types)
        return self.pp_output_layer.get_lambda_0(enc_output)
