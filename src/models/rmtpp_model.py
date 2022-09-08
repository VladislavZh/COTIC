from typing import Any, List

import torch
import numpy as np
from pytorch_lightning import LightningModule

from src.utils.metrics import MetricsCore


class RMTPPModule(LightningModule):
    """
    Base event sequence lightning module
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
        metrics: MetricsCore,
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

        self.train_metrics = metrics
        self.val_metrics = metrics.copy_empty()
        self.test_metrics = metrics.copy_empty()
        # losses
        self.class_weights = np.ones(self.num_class)
        self.event_criterion = torch.nn.CrossEntropyLoss(
            weight=torch.FloatTensor(self.class_weights)
        )
        self.intensity_w = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float))
        self.intensity_b = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float))
        self.time_criterion = self.RMTPPLoss

        self.alpha = alpha
        self.lr = lr

    def RMTPPLoss(self, pred, gold):
        loss = torch.mean(
            pred
            + self.intensity_w * gold
            + self.intensity_b
            + (
                torch.exp(pred + self.intensity_b)
                - torch.exp(pred + self.intensity_w * gold + self.intensity_b)
            )
            / self.intensity_w
        )
        return -1 * loss

    def forward(self, input_time, input_events):

        event_embedding = self.embedding(input_events)
        event_embedding = self.emb_dropout(event_embedding)
        lstm_input = torch.cat((event_embedding, input_time.unsqueeze(-1)), dim=-1)
        hidden_state, _ = self.lstm(lstm_input)

        # hidden_state = torch.cat((hidden_state, input_time.unsqueeze(-1)), dim=-1)
        mlp_output = torch.tanh(self.mlp(hidden_state[:, -1, :]))
        mlp_output = self.mlp_dropout(mlp_output)
        event_logits = self.event_linear(mlp_output)
        time_logits = self.time_linear(mlp_output)

        return time_logits, event_logits

    def step(self, batch: Any):

        time_tensor, event_tensor = batch
        time_input, time_target = time_tensor[:, :-1], time_tensor[:, -1]
        event_input, event_target = event_tensor[:, :-1], event_tensor[:, -1]
        time_logits, event_logits = self.forward(time_input, event_input)
        loss_time = self.time_criterion(time_logits.view(-1), time_target.view(-1))
        loss_event = self.event_criterion(
            event_logits.view(-1, self.num_class), event_target.view(-1)
        )
        loss = self.alpha * loss_time + loss_event

        return loss, loss_time, loss_event

    def predict_step(self, batch: Any):

        time_tensor, event_tensor = batch
        time_input, time_target = time_tensor[:, :-1], time_tensor[:, -1]
        event_input, event_target = event_tensor[:, :-1], event_tensor[:, -1]
        time_logits, event_logits = self.forward(time_input, event_input)
        event_pred = np.argmax(event_logits.detach().cpu().numpy(), axis=-1)
        time_pred = time_logits.detach().cpu().numpy()

        return event_pred, time_pred

    def training_step(self, batch: Any, batch_idx: int):
        loss, loss_time, loss_event = self.step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        ll, return_time_metric, event_type_metric = self.train_metrics.compute_metrics()
        self.train_metrics.clear_values()

        self.log(
            "train/log_likelihood", ll, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/return_time_metric",
            return_time_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/event_type_metric",
            event_type_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def validation_step(self, batch: Any, batch_idx: int):
        loss, loss_time, loss_event = self.step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        ll, return_time_metric, event_type_metric = self.val_metrics.compute_metrics()
        self.val_metrics.clear_values()

        self.log("val/log_likelihood", ll, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/return_time_metric",
            return_time_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/event_type_metric",
            event_type_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch: Any, batch_idx: int):
        loss, loss_time, loss_event = self.step(batch)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

    def test_epoch_end(self, outputs: List[Any]):
        ll, return_time_metric, event_type_metric = self.test_metrics.compute_metrics()
        self.test_metrics.clear_values()

        self.log("test/log_likelihood", ll, on_step=False, on_epoch=True)
        self.log(
            "test/return_time_metric", return_time_metric, on_step=False, on_epoch=True
        )
        self.log(
            "test/event_type_metric", event_type_metric, on_step=False, on_epoch=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda x: x.requires_grad, self.parameters()),
            self.lr,
            betas=(0.9, 0.999),
            eps=1e-05,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

        return [optimizer], [lr_scheduler]
