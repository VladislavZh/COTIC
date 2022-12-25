from typing import Any, List

import torch
import numpy as np
from pytorch_lightning import LightningModule

from src.utils.metrics.scores import RocAuc, MAE, Accuracy


class RMTPPModule(LightningModule):
    """
    Recurrent Marked Temporal Point Process (Du et al. 2016) lightning module
    """

    def __init__(
        self,
        num_class: int,
        emb_dim: int,
        hid_dim: int,
        mlp_dim: int,
        dropout: float,
        alpha: float,
        lr: float,
        weight_decay: float,
    ):

        super().__init__()
        self.num_class = num_class
        self.embedding = torch.nn.Embedding(
            num_embeddings=num_class, embedding_dim=emb_dim
        )
        self.emb_dropout = torch.nn.Dropout(p=dropout)
        self.lstm = torch.nn.LSTM(
            input_size=emb_dim + 1,
            hidden_size=hid_dim,
            batch_first=True,
            bidirectional=False,
        )
        self.mlp = torch.nn.Linear(in_features=hid_dim, out_features=mlp_dim)
        self.mlp_dropout = torch.nn.Dropout(p=dropout)
        self.event_linear = torch.nn.Linear(in_features=mlp_dim, out_features=num_class)
        self.time_linear = torch.nn.Linear(in_features=mlp_dim, out_features=1)

        self.intensity_w = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float))
        self.intensity_b = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float))
        self.alpha = alpha
        self.lr = lr
        self.weight_decay = weight_decay

        self.time_criterion = self.RMTPPLoss
        #self.class_weights = np.ones(self.num_class)
        self.event_criterion = torch.nn.CrossEntropyLoss()
        #    weight=torch.FloatTensor(self.class_weights)
        #)

        self.time_metric = MAE
        self.event_metric = Accuracy

    def RMTPPLoss(self, pred, target):
        loss = torch.mean(
            pred
            + self.intensity_w * target
            + self.intensity_b
            + (
                torch.exp(pred + self.intensity_b)
                - torch.exp(pred + self.intensity_w * target + self.intensity_b)
            )
            / self.intensity_w
        )
        return -1 * loss

    def forward(self, input_time, input_events):
        
        event_embedding = self.embedding(input_events)
        event_embedding = self.emb_dropout(event_embedding)
        lstm_input = torch.cat((event_embedding, input_time.unsqueeze(-1)), dim=-1)
        hidden_state, _ = self.lstm(lstm_input)

        mlp_output = torch.tanh(self.mlp(hidden_state[:, -1, :]))
        mlp_output = self.mlp_dropout(mlp_output)
        event_logits = self.event_linear(mlp_output)
        time_logits = self.time_linear(mlp_output)

        return time_logits, event_logits

    def step(self, batch: Any):

        time_input, event_input, time_target, event_target = batch
        time_logits, event_logits = self.forward(time_input, event_input)
        loss_time = self.time_criterion(time_logits.view(-1), time_target.view(-1))
        loss_event = self.event_criterion(
            event_logits.view(-1, self.num_class), event_target.view(-1)
        )
        loss = -1 * self.alpha * loss_time - loss_event
        event_pred = event_logits
        time_pred = time_logits

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
            predicted_events = torch.cat([predicted_events, outputs[i]["event_pred"]], dim=0)
            gt_events = torch.cat([gt_events, outputs[i]["event_target"]], dim=0)
            predicted_times = torch.cat([predicted_times, outputs[i]["time_pred"]], dim=0)
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
            predicted_events = torch.cat([predicted_events, outputs[i]["event_pred"]], dim=0)
            gt_events = torch.cat([gt_events, outputs[i]["event_target"]], dim=0)
            predicted_times = torch.cat([predicted_times, outputs[i]["time_pred"]], dim=0)
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
            predicted_events = torch.cat([predicted_events, outputs[i]["event_pred"]], dim=0)
            gt_events = torch.cat([gt_events, outputs[i]["event_target"]], dim=0)
            predicted_times = torch.cat([predicted_times, outputs[i]["time_pred"]], dim=0)
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
