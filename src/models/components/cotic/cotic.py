import torch
import torch.nn as nn

from .cotic_layers import ContConv1d, ContConv1dSim


class COTIC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        nb_filters: int,
        nb_layers: int,
        num_types: int,
        kernel: nn.Module,
        head: nn.Module,
    ) -> None:
        super().__init__()
        self.event_emb = nn.Embedding(num_types + 2, in_channels, padding_idx=0)

        self.in_channels = [in_channels] + [nb_filters] * nb_layers
        include_zero_lag = [True] + [True] * nb_layers
        self.dilation_factors = [2**i for i in range(0, nb_layers)]

        self.num_types = num_types
        self.nb_layers = nb_layers
        self.nb_filters = nb_filters

        self.convs = nn.ModuleList(
            [
                ContConv1d(
                    kernel.recreate(self.in_channels[i]),
                    kernel_size,
                    self.in_channels[i],
                    nb_filters,
                    self.dilation_factors[i],
                    include_zero_lag[i],
                )
                for i in range(nb_layers)
            ]
        )

        self.final_list = nn.ModuleList(
            [
                ContConv1dSim(
                    kernel.recreate(self.nb_filters), 1, nb_filters, nb_filters
                ),
                nn.LeakyReLU(0.1),
                nn.Linear(nb_filters, num_types),
                nn.Softplus(100),
            ]
        )

        self.head = head

    @staticmethod
    def __add_bos(event_times, event_types):
        bos_event_times = torch.cat([torch.zeros(event_times.shape[0], 1, device=event_times.device), event_times],
                                    dim=1)
        max_event_type = torch.max(event_types) + 1
        bos_event_types = torch.cat(
            [torch.full((event_types.shape[0], 1), max_event_type, dtype=torch.long, device=event_types.device),
             event_types], dim=1)
        return bos_event_times, bos_event_types

    def forward(
        self, event_times: torch.Tensor, event_types: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass that computes self.convs and return encoder output

        args:
            event_times - torch.Tensor, shape = (bs, L) event times
            event_types - torch.Tensor, shape = (bs, L) event types
            lengths - torch.Tensor, shape = (bs,) sequence lengths
        """
        event_times, event_types = self.__add_bos(event_times, event_types)

        non_pad_mask = event_types.ne(0)

        enc_output = self.event_emb(event_types)

        for conv in self.convs:
            enc_output = torch.nn.functional.leaky_relu(
                conv(event_times, enc_output, non_pad_mask), 0.1
            )

        return enc_output, self.head(enc_output.detach())

    def final(self, times, true_times, true_features, non_pad_mask, sim_size):
        out = self.final_list[0](times, true_times, true_features, non_pad_mask, sim_size)
        for layer in self.final_list[1:]:
            out = layer(out)
        return out
