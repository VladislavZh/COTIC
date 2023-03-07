import numpy as np

import torch
from pytorch_lightning import LightningModule

from typing import Union, Tuple, Any, List
from scipy import integrate

from src.utils.metrics import MetricsCore


def differentiate(in_times: torch.Tensor) -> torch.Tensor:
    """Differentiate sequence with event times: compute time increments.

    args:
        in_times - sequence of event times to be differentiated

    returns:
        dt - sequence of time increments
    """
    diff = torch.diff(in_times)
    zeros = torch.zeros(in_times.shape[0]).unsqueeze(dim=1).to(in_times.device)
    dt = torch.hstack((zeros, diff))
    return dt


class CCNNBaselineMetrics(MetricsCore):
    def __init__(
        self,
        return_time_metric: Any,
        event_type_metric: Any,
        reductions: dict = {"log_likelihood": "mean", "type": "sum", "time": "mean"},
    ) -> None:
        """Initialize WaveNet metrics core."""
        super().__init__(return_time_metric, event_type_metric)
        self.reductions = reductions

        # this model has no extra heads with additional type_loss and time_loss
        self.type_loss_func = None
        self.return_time_loss_func = None

    @staticmethod
    def get_return_time_target(inputs: Union[Tuple, torch.Tensor]) -> torch.Tensor:
        """
        Takes input batch and returns the corresponding return time targets as 1d Tensor

        args:
            inputs - Tuple or torch.Tensor, batch received from the dataloader

        return:
            return_time_target - torch.Tensor, 1d Tensor with return time targets
        """
        event_times, event_types = inputs
        dt_batch = differentiate(event_times)[event_types > 0]
        return dt_batch

    @staticmethod
    def get_event_type_target(inputs: Union[Tuple, torch.Tensor]) -> torch.Tensor:
        """
        Takes input batch and returns the corresponding event type targets as 1d Tensor

        args:
            inputs - Tuple or torch.Tensor, batch received from the dataloader

        return:
            event_type_target - torch.Tensor,  1d Tensor with event type target
        """
        event_type = inputs[1][:, 1:]
        mask = inputs[1].ne(0)[:, 1:]
        return event_type[mask]

    @staticmethod
    def compute_return_time_seq(
        lambda_0: np.ndarray,
        w_t: np.ndarray,
    ) -> List[float]:
        """Estimetate return times analytically (as the integral) for a single sequence.

        args:
            lambda_0 - intensity values to be integrated
            w_t - scalar weight coefficient (trainable param in pp_output_layer)

        return:
            predictions - list of estimated return times for a sequence
        """

        def integrand(tao: np.ndarray, lambda_0: np.ndarray) -> np.ndarray:
            """Form of the function to be integrated.

            args:
                tao - time variable
                lambda_0 - intensity values to be integrated
            """
            if w_t != 0:
                res = (
                    tao
                    * lambda_0
                    * np.exp(w_t * tao)
                    * np.exp(lambda_0 / w_t * (1 - np.exp(w_t * tao)))
                )
                if np.isnan(res):
                    return 0
                return res
            else:
                res = tao * lambda_0 * np.exp(-tao * lambda_0)
                if np.isnan(res):
                    return 0
                return res

        predictions = list()
        for l in lambda_0:
            expected_dt = integrate.quad(integrand, 0, np.inf, args=(l,))[0]
            predictions.append(expected_dt)

        return predictions

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
        event_times_batch, event_types_batch = inputs

        if isinstance(event_times_batch, (np.ndarray, np.generic)):
            event_times_batch = torch.from_numpy(event_times_batch)

        lambda_0_batch = pl_module.net.get_lambda_0(
            event_times_batch, event_types_batch
        ).squeeze()[:, 1:]

        return_time_list = []
        for l, types in zip(lambda_0_batch, event_types_batch):
            l = l[types > 0]  # [1:]
            return_time = CCNNBaselineMetrics.compute_return_time_seq(
                l.detach().cpu().numpy(),
                pl_module.net.pp_output_layer.w_t.detach().cpu().numpy(),
            )
            return_time_list.extend(return_time)

        # vec_compute_return_time_seq = np.vectorize(
        #    CCNNBaselineMetrics.compute_return_time_seq,
        #    excluded=["w_t"],
        # )

        # return_time_list = vec_compute_return_time_seq(
        #    lambda_0=lambda_0_batch.detach().cpu().numpy(),
        #    w_t=pl_module.net.pp_output_layer.w_t.detach().cpu().numpy(),
        #    event_types_seq=event_types_batch.detach().cpu().numpy(),
        # )

        return_time_predicted = torch.Tensor(return_time_list)
        return return_time_predicted

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
            outputs - Tuple or torch.Tensor, model output - (multinomial, log_likelihood)

        return:
            event_type_predicted - torch.Tensor, 2d Tensor with event type unnormalized predictions
        """
        event_type_prediction = outputs[0][:, 1:-1, :]
        mask = inputs[1].ne(0)[:, 1:]
        return event_type_prediction[mask, :]

    def compute_log_likelihood_per_event(
        self,
        pl_module: LightningModule,
        inputs: Union[Tuple, torch.Tensor],
        outputs: Union[Tuple, torch.Tensor],
    ) -> torch.Tensor:
        """Placeholder with NaN values of shape batch_size."""
        bs = inputs[0].shape[0]
        return torch.ones(bs) * torch.nan

    def pp_likelihood_loss(
        self,
        labels: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log-likelihood loss for Baseline CCNN model.

        args:
            labels - true event types
            output - model's output (a tuple of P(y|h) and f(t))

        return:
            loss - log-likelihood loss, i.e. \sum_i \sum_j (log(P(y|h) + log(f(t)))
        """
        multinomial, log_likelihood = output
        likelihood = torch.exp(log_likelihood)

        gather_idx = torch.where(labels > 0)

        # use small constant to avoid division by zero
        OVERFLOW_CONSTANT = (
            torch.Tensor([1e-32]).to(torch.float32).to(multinomial.device)
        )

        log_p = torch.log(torch.max(multinomial[gather_idx], OVERFLOW_CONSTANT))
        log_f = torch.log(torch.max(likelihood[gather_idx], OVERFLOW_CONSTANT))

        # no reduction here
        loss = -1 * (log_p + log_f)
        return loss

    def type_loss(self, prediction: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        """Placeholder with constant zero value. No additional type loss for this model."""
        return 0.0

    def time_loss(
        self,
        prediction: torch.Tensor,
        event_time: torch.Tensor,
        event_type: torch.Tensor,
    ) -> torch.Tensor:
        """Placeholder with constant zero value. No additional time loss for this model."""
        return 0.0

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
            outputs - Tuple or torch.Tensor, model output: Tuple (multinomial, likelihood)

        return:
            loss - log-likelihood loss for backpropagation
        """
        labels = inputs[1]
        pp_likelihood_loss = self.pp_likelihood_loss(labels, outputs)

        if self.reductions["log_likelihood"] not in ["mean", "sum"]:
            raise ValueError("log_likelihood reduction not in 'mean', 'sum'")
        if self.reductions["log_likelihood"] == "mean":
            ll_loss = pp_likelihood_loss.mean()
        else:
            ll_loss = pp_likelihood_loss.sum()

        # return zeros as the second item, no 'time_loss' and 'type_loss' for this model
        return ll_loss, 0.0
