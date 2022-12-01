import torch.nn as nn
import torch

from .encoder import Encoder
from .final_layers import RNN_layers, Predictor


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1, rnn=True):
        super().__init__()

        self.encoder = Encoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        self.num_types = num_types

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn_flag = rnn
        if rnn:
            self.rnn = RNN_layers(d_model, d_rnn)

        # prediction of next time stamp
        self.time_predictor = Predictor(d_model, 1)

        # prediction of next event type
        self.type_predictor = Predictor(d_model, num_types)

    @staticmethod
    def get_non_pad_mask(seq):
        """ Get the non-padding positions. """

        assert seq.dim() == 2
        return seq.ne(0).type(torch.float).unsqueeze(-1)

    def forward(self, event_time, event_type):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

        non_pad_mask = self.get_non_pad_mask(event_type)

        enc_output = self.encoder(event_type, event_time, non_pad_mask)

        if self.rnn_flag:
            enc_output = self.rnn(enc_output, non_pad_mask)

        time_prediction = self.time_predictor(enc_output.detach(), non_pad_mask)

        type_prediction = self.type_predictor(enc_output.detach(), non_pad_mask)

        return enc_output, (type_prediction, time_prediction)

    @staticmethod
    def softplus(x, beta):
        # hard thresholding at 20
        temp = beta * x
        temp[torch.abs(temp) > 20] = 20
        return 1.0 / beta * torch.log(1 + torch.exp(temp))

    def final(self, times, true_times, true_features, non_pad_mask, sim_size):
        bs, L = true_times.shape
        times = times[:,1:].reshape(bs, L-1, sim_size+1)
        times = times - true_times[:,:-1].unsqueeze(2)
        times /= (true_times[:, :-1] + 1).unsqueeze(2)

        temp_hid = self.linear(true_features)[:, :-1, :].unsqueeze(2)
        temp_hid = torch.concat([torch.zeros(1,1,1,self.num_types, device=temp_hid.device),temp_hid], axis=1)
        all_lambda = self.softplus(temp_hid + self.alpha * times.unsqueeze(3), self.beta)
        all_lambda = all_lambda.reshape(bs, -1, self.num_types)
        all_lambda = torch.concat([torch.zeros(1,1,self.num_types, device=all_lambda.device),all_lambda], axis=1)

        return all_lambda
