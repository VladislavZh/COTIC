from typing import Any, List

import math
import torch
import numpy as np
import numba
from pytorch_lightning import LightningModule

from src.utils.metrics.scores import MAE, Accuracy
from src.models.components.baselines.rocket.kernel_utils import (
    generate_kernels,
    apply_kernels,
)


class RocketModule(LightningModule):
    """
    Sequential model with random kernels, inspired by
    "ROCKET: Exceptionally fast and accurate time series classification "
    (Dempster et al., 2020)
    """

    def __init__(
        self,
        num_class: int,
        num_kernels: int,
        max_len: int,
        emb_dim: int,
        dropout: float,
        alpha: float,
        lr: float,
        weight_decay: float,
    ):

        super().__init__()
        self.num_class = num_class
        self.num_kernels = num_kernels
        self.max_len = max_len
        self.embedding = torch.nn.Embedding(
            num_embeddings=num_class + 1, embedding_dim=emb_dim, padding_idx=0
        )
        self.emb_dropout = torch.nn.Dropout(p=dropout)
        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / emb_dim) for i in range(emb_dim)]
        )

        # (
        #     self.kweights,
        #     self.klengths,
        #     self.kbiases,
        #     self.kdilations,
        #     self.kpaddings,
        # )
        self.rocket_kernels = generate_kernels(self.max_len, self.num_kernels)

        self.event_linear = torch.nn.Linear(
            in_features=2 * num_kernels * emb_dim, out_features=num_class
        )
        self.time_linear = torch.nn.Linear(
            in_features=2 * num_kernels * emb_dim, out_features=1
        )

        self.alpha = alpha
        self.lr = lr
        self.weight_decay = weight_decay

        self.time_criterion = torch.nn.MSELoss()
        self.class_weights = np.ones(self.num_class)
        self.event_criterion = torch.nn.CrossEntropyLoss(
            weight=torch.FloatTensor(self.class_weights)
        )

        self.time_metric = MAE
        self.event_metric = Accuracy

    def temporal_enc(self, time):
        """
        Input: batch * seq_len
        Output: batch * seq_len * emb_dim
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result

    def forward(self, input_times, input_events):

        events_embedding = self.embedding(input_events)
        events_embedding = self.emb_dropout(events_embedding)
        times_embedding = self.temporal_enc(input_times)
        events_embedding = torch.reshape(
            events_embedding,
            (
                events_embedding.shape[0] * events_embedding.shape[2],
                events_embedding.shape[1],
            ),
        )
        times_embedding = torch.reshape(
            times_embedding,
            (
                times_embedding.shape[0] * times_embedding.shape[2],
                times_embedding.shape[1],
            ),
        )
        events_feature = apply_kernels(events_embedding, self.rocket_kernels)
        times_feature = apply_kernels(times_embedding, self.rocket_kernels)
        events_feature = torch.reshape(
            events_feature,
            (input_events.shape[0], -1),
        )
        times_feature = torch.reshape(
            times_feature,
            (input_times.shape[0], -1),
        )
        
        events_pred = self.event_linear(events_feature)
        times_pred = self.time_linear(times_feature)

        return times_pred, events_pred

    def step(self, batch: Any):

        time_input, event_input, time_target, event_target = batch
        time_pred, event_pred = self.forward(time_input, event_input)
        loss_time = self.time_criterion(time_pred.view(-1), time_target.view(-1))
        loss_event = self.event_criterion(
            event_pred.view(-1, self.num_class), event_target.view(-1)
        )
        loss = self.alpha * loss_time + loss_event

        return (
            loss,
            loss_time,
            loss_event,
            event_target,
            time_target,
            event_pred,
            time_pred,
        )

    def training_step(self, batch: Any, batch_idx: int):
        (
            loss,
            loss_time,
            loss_event,
            event_target,
            time_target,
            event_pred,
            time_pred,
        ) = self.step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "train/loss_time", loss_time, on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            "train/loss_event", loss_event, on_step=False, on_epoch=True, prog_bar=False
        )
        return {
            "loss": loss,
            "loss_time": loss_time,
            "loss_event": loss_event,
            "event_target": event_target,
            "time_target": time_target,
            "event_pred": event_pred,
            "time_pred": time_pred,
        }

    def training_epoch_end(self, outputs: List[Any]):

        predicted_events = outputs[0]["event_pred"]
        gt_events = outputs[0]["event_target"]
        predicted_times = outputs[0]["time_pred"]
        gt_times = outputs[0]["time_target"]
        for i in range(1, len(outputs)):
            predicted_events = torch.cat(
                [predicted_events, outputs[i]["event_pred"]], dim=0
            )
            gt_events = torch.cat([gt_events, outputs[i]["event_target"]], dim=0)
            predicted_times = torch.cat(
                [predicted_times, outputs[i]["time_pred"]], dim=0
            )
            gt_times = torch.cat([gt_times, outputs[i]["time_target"]], dim=0)

        event_metric_train = self.event_metric(predicted_events, gt_events)
        time_metric_train = self.time_metric(predicted_times, gt_times)
        self.log("train/accuracy", event_metric_train, prog_bar=True)
        self.log("train/mae", time_metric_train, prog_bar=True)

    def validation_step(self, batch: Any, batch_idx: int):
        (
            loss,
            loss_time,
            loss_event,
            event_target,
            time_target,
            event_pred,
            time_pred,
        ) = self.step(batch)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "val/loss_time", loss_time, on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            "val/loss_event", loss_event, on_step=False, on_epoch=True, prog_bar=False
        )

        return {
            "loss": loss,
            "loss_time": loss_time,
            "loss_event": loss_event,
            "event_target": event_target,
            "time_target": time_target,
            "event_pred": event_pred,
            "time_pred": time_pred,
        }

    def validation_epoch_end(self, outputs: List[Any]):
        predicted_events = outputs[0]["event_pred"]
        gt_events = outputs[0]["event_target"]
        predicted_times = outputs[0]["time_pred"]
        gt_times = outputs[0]["time_target"]
        for i in range(1, len(outputs)):
            predicted_events = torch.cat(
                [predicted_events, outputs[i]["event_pred"]], dim=0
            )
            gt_events = torch.cat([gt_events, outputs[i]["event_target"]], dim=0)
            predicted_times = torch.cat(
                [predicted_times, outputs[i]["time_pred"]], dim=0
            )
            gt_times = torch.cat([gt_times, outputs[i]["time_target"]], dim=0)

        event_metric_val = self.event_metric(predicted_events, gt_events)
        time_metric_val = self.time_metric(predicted_times, gt_times)
        self.log("val/accuracy", event_metric_val, prog_bar=True)
        self.log("val/mae", time_metric_val, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        (
            loss,
            loss_time,
            loss_event,
            event_target,
            time_target,
            event_pred,
            time_pred,
        ) = self.step(batch)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "test/loss_time", loss_time, on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            "test/loss_event", loss_event, on_step=False, on_epoch=True, prog_bar=False
        )

        return {
            "loss": loss,
            "loss_time": loss_time,
            "loss_event": loss_event,
            "event_target": event_target,
            "time_target": time_target,
            "event_pred": event_pred,
            "time_pred": time_pred,
        }

    def test_epoch_end(self, outputs: List[Any]):
        predicted_events = outputs[0]["event_pred"]
        gt_events = outputs[0]["event_target"]
        predicted_times = outputs[0]["time_pred"]
        gt_times = outputs[0]["time_target"]
        for i in range(1, len(outputs)):
            predicted_events = torch.cat(
                [predicted_events, outputs[i]["event_pred"]], dim=0
            )
            gt_events = torch.cat([gt_events, outputs[i]["event_target"]], dim=0)
            predicted_times = torch.cat(
                [predicted_times, outputs[i]["time_pred"]], dim=0
            )
            gt_times = torch.cat([gt_times, outputs[i]["time_target"]], dim=0)

        event_metric_test = self.event_metric(predicted_events, gt_events)
        time_metric_test = self.time_metric(predicted_times, gt_times)
        self.log("test/accuracy", event_metric_test, prog_bar=True)
        self.log("test/mae", time_metric_test, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda x: x.requires_grad, self.parameters()),
            self.lr,
            betas=(0.9, 0.999),
            eps=1e-05,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

        return [optimizer], [lr_scheduler]
