import torch
from pytorch_lightning import LightningModule
import math

from src.utils.metrics import MetricsCore
from typing import Union, Tuple


class THPMetrics(MetricsCore):
    def __init__(self, return_time_metric, event_type_metric, scale_time_loss=100):
        super().__init__(return_time_metric, event_type_metric)
        self.type_loss_func = torch.nn.CrossEntropyLoss(
            ignore_index=-1, reduction="none"
        )
        self.scale_time_loss = scale_time_loss

    @staticmethod
    def get_return_time_target(inputs: Union[Tuple, torch.Tensor]) -> torch.Tensor:
        """
        Takes input batch and returns the corresponding return time targets as 1d Tensor

        args:
            inputs - Tuple or torch.Tensor, batch received from the dataloader

        return:
            return_time_target - torch.Tensor, 1d Tensor with return time targets
        """
        event_time = inputs[0]
        return_time = event_time[:, 1:] - event_time[:, :-1]
        mask = inputs[1].ne(0)[:, 1:]
        return return_time[mask]

    @staticmethod
    def get_event_type_target(inputs: Union[Tuple, torch.Tensor]) -> torch.Tensor:
        """
        Takes input batch and returns the corresponding event type targets as 1d Tensor

        args:
            inputs - Tuple or torch.Tensor, batch received from the dataloader

        return:
            event_type_target - torch.Tensor, 1d Tensor with event type targets
        """
        event_type = inputs[1][:, 1:]
        mask = inputs[1].ne(0)[:, 1:]
        return event_type[mask]

    @staticmethod
    def get_return_time_predicted(
        pl_module: LightningModule,
        inputs: Union[Tuple, torch.Tensor],
        outputs: Union[Tuple, torch.Tensor],
    ) -> torch.Tensor:
        """
        Takes lighning model, input batch and model outputs, returns the corresponding predicted return times as 1d Tensor

        args:
            pl_module - LightningModule, training lightning model
            inputs - Tuple or torch.Tensor, batch received from the dataloader
            outputs - Tuple or torch.Tensor, model output

        return:
            return_time_predicted - torch.Tensor, 1d Tensor with return time prediction
        """
        return_time_prediction = outputs[1][1].squeeze_(-1)[:, :-1]
        mask = inputs[1].ne(0)[:, 1:]
        return return_time_prediction[mask]

    @staticmethod
    def get_event_type_predicted(
        pl_module: LightningModule,
        inputs: Union[Tuple, torch.Tensor],
        outputs: Union[Tuple, torch.Tensor],
    ) -> torch.Tensor:
        """
        Takes lighning model, input batch and model outputs, returns the corresponding predicted event types as 1d Tensor

        args:
            pl_module - LightningModule, training lightning model
            inputs - Tuple or torch.Tensor, batch received from the dataloader
            outputs - Tuple or torch.Tensor, model output

        return:
            event_type_predicted - torch.Tensor, 2d Tensor with event type unnormalized predictions
        """
        event_type_prediction = outputs[1][0][:, :-1, :]
        mask = inputs[1].ne(0)[:, 1:]
        return event_type_prediction[mask, :]

    @staticmethod
    def softplus(x, beta):
        # hard thresholding at 20
        temp = beta * x
        temp[temp > 20] = 20
        return 1.0 / beta * torch.log(1 + torch.exp(temp))

    @staticmethod
    def compute_event(type_lambda, non_pad_mask):
        """Log-likelihood of events."""

        # add 1e-9 in case some events have 0 likelihood
        type_lambda += math.pow(10, -9)
        type_lambda.masked_fill_(~non_pad_mask.bool(), 1.0)

        result = torch.log(type_lambda)
        return result

    def compute_integral_unbiased(self, model, data, time, non_pad_mask, type_mask):
        """Log-likelihood of non-events, using Monte Carlo integration."""

        num_samples = 100

        diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
        temp_time = diff_time.unsqueeze(2) * torch.rand(
            [*diff_time.size(), num_samples], device=data.device
        )
        temp_time /= (time[:, :-1] + 1).unsqueeze(2)

        temp_hid = model.linear(data)[:, 1:, :]
        temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)

        all_lambda = self.softplus(temp_hid + model.alpha * temp_time, model.beta)
        all_lambda = torch.sum(all_lambda, dim=2) / num_samples

        unbiased_integral = all_lambda * diff_time
        return unbiased_integral

    def event_and_non_event_log_likelihood(
        self,
        pl_module: LightningModule,
        enc_output: torch.Tensor,
        event_time: torch.Tensor,
        event_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes log of the intensity and the integral
        """

        non_pad_mask = event_type.ne(0).type(torch.float)

        type_mask = torch.zeros(
            [*event_type.size(), pl_module.net.num_types], device=enc_output.device
        )
        for i in range(pl_module.net.num_types):
            type_mask[:, :, i] = (event_type == i + 1).bool().to(enc_output.device)

        all_hid = pl_module.net.linear(enc_output)
        all_lambda = self.softplus(all_hid, pl_module.net.beta)
        type_lambda = torch.sum(all_lambda * type_mask, dim=2)  # shape = (bs, L)

        # event log-likelihood
        event_ll = self.compute_event(type_lambda, non_pad_mask)
        event_ll = torch.sum(event_ll, dim=-1)

        # non-event log-likelihood, MC integration
        non_event_ll = self.compute_integral_unbiased(
            pl_module.net, enc_output, event_time, non_pad_mask, type_mask
        )
        non_event_ll = torch.sum(non_event_ll, dim=-1)

        return event_ll, non_event_ll

    def compute_log_likelihood_per_event(
        self,
        pl_module: LightningModule,
        inputs: Union[Tuple, torch.Tensor],
        outputs: Union[Tuple, torch.Tensor],
    ) -> torch.Tensor:
        """
        Takes lighning model, input batch and model outputs, returns the corresponding log likelihood per event for each sequence in the batch as 1d Tensor of shape (bs,),
        one can use self.step_[return_time_target/event_type_target/return_time_predicted/event_type_predicted] if needed

        args:
            pl_module - LightningModule, training lightning model
            inputs - Tuple or torch.Tensor, batch received from the dataloader
            outputs - Tuple or torch.Tensor, model output

        return:
            log_likelihood_per_seq - torch.Tensor, 1d Tensor with log likelihood per event prediction, shape = (bs,)
        """
        event_ll, non_event_ll = self.event_and_non_event_log_likelihood(
            pl_module, outputs[0], inputs[0], inputs[1]
        )
        lengths = torch.sum(inputs[1].ne(0).type(torch.float), dim=1)
        results = (event_ll - non_event_ll) / lengths
        return results

    def type_loss(self, prediction, types):
        """Event prediction loss, cross entropy."""

        # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
        truth = types[:, 1:] - 1
        prediction = prediction[:, :-1, :]

        loss = self.type_loss_func(prediction.transpose(1, 2), truth)

        loss = torch.sum(loss)
        return loss

    @staticmethod
    def time_loss(prediction, event_time):
        """Time prediction loss."""

        prediction.squeeze_(-1)

        true = event_time[:, 1:] - event_time[:, :-1]
        prediction = prediction[:, :-1]

        # event time gap prediction
        diff = prediction - true
        se = torch.sum(diff * diff)
        return se

    def compute_loss(
        self,
        pl_module: LightningModule,
        inputs: Union[Tuple, torch.Tensor],
        outputs: Union[Tuple, torch.Tensor],
    ) -> torch.Tensor:
        """
        Takes lighning model, input batch and model outputs, returns the corresponding loss for backpropagation,
        one can use self.step_[return_time_target/event_type_target/return_time_predicted/event_type_predicted/ll_per_event] if needed

        args:
            pl_module - LightningModule, training lightning model
            inputs - Tuple or torch.Tensor, batch received from the dataloader
            outputs - Tuple or torch.Tensor, model output

        return:
            loss - torch.Tensor, loss for backpropagation
        """
        event_ll, non_event_ll = self.event_and_non_event_log_likelihood(
            pl_module, outputs[0], inputs[0], inputs[1]
        )
        ll_loss = -torch.mean(event_ll - non_event_ll)
        type_loss = self.type_loss(outputs[1][0], inputs[1])
        time_loss = self.time_loss(outputs[1][1], inputs[0])

        return ll_loss + type_loss + time_loss / self.scale_time_loss
