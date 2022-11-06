from typing import Any, List

import torch
import numpy as np
from pytorch_lightning import LightningModule
from scipy.integrate import trapz
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.utils.metrics.scores import RocAuc, MAE, Accuracy


class RNNSMModule(LightningModule):
    """
    Recurrent Neural Network Survival Model: Predicting Web User Return Time (Grob et al. 2018) lightning module
    """

    def __init__(
        self,
        hid_dim: int,
        emb_dims: List[int],
        cat_sizes: List[int],
        input_size: int,
        lstm_hid_dim: int,
        n_num_feats: int,
        padding: int,
        w_trainable: bool,
        w: float,
        dropout: float,
        time_scale: float,
        prediction_start: float,
        integration_end: float,
        lr: float,
        weight_decay: float,
    ):

        super().__init__()
        self.hid_dim = hid_dim
        self.emb_dims = emb_dims
        self.cat_sizes = cat_sizes
        self.LSTM = torch.nn.LSTM(input_size, lstm_hid_dim, batch_first=True)
        self.embeddings = torch.nn.ModuleList([torch.nn.Embedding(cat_size + 1, emb_dim, padding_idx=padding) \
                                         for cat_size, emb_dim in zip(cat_sizes, emb_dims)])
        
        total_emb_length = sum(emb_dims)
        self.input_dense = torch.nn.Linear(total_emb_length + n_num_feats, input_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.output_dense = torch.nn.Linear(hid_dim, 1, bias=False)
        self.hidden = torch.nn.Linear(lstm_hid_dim, hid_dim)

        if w_trainable:
            self.w = torch.nn.Parameter(torch.FloatTensor([0.1]))
        else:
            self.w = w

        self.time_scale = time_scale
        self.prediction_start = prediction_start
        self.integration_end = integration_end

        self.time_criterion = self.RMTPPLoss
        self.class_weights = np.ones(self.num_class)
        self.labels = np.arange(1, self.num_class + 1)
        self.event_criterion = torch.nn.CrossEntropyLoss(
            weight=torch.FloatTensor(self.class_weights)
        )

        self.time_metric = MAE
        self.event_metric = Accuracy

    def _s_t(self, last_o_j, deltas):
        out = torch.exp(torch.exp(last_o_j) / self.w - \
                        torch.exp(last_o_j + self.w * deltas) / self.w)
        return out.squeeze()

    def rnnsm_loss(self, deltas, non_pad_mask, ret_mask, o_j):
        deltas_scaled = deltas * self.time_scale
        p = o_j + self.w * deltas_scaled
        common_term = -(torch.exp(o_j) - torch.exp(p)) / self.w * non_pad_mask
        common_term = common_term.sum() / (non_pad_mask).sum()
        ret_term = (-p * ret_mask).sum() / ret_mask.sum()
        return common_term + ret_term

    def forward(self, cat_feats, num_feats, lengths):
        x = torch.zeros((cat_feats.size(0), cat_feats.size(1), 0)).to(cat_feats.device)
        for i, emb in enumerate(self.embeddings):
            x = torch.cat([x, emb(cat_feats[:, :, i])], axis=-1)
        x = torch.cat([x, num_feats], axis=-1)
        x = self.dropout(torch.tanh(self.input_dense(x)))
        x = pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
        h_j, _ = self.lstm(x)
        h_j, _ = pad_packed_sequence(h_j, batch_first=True)
        h_j = self.dropout(torch.tanh(self.hidden(h_j)))
        o_j = self.output_dense(h_j).squeeze()
        return o_j
        
    def predict(self, o_j, t_j, lengths):
        with torch.no_grad():
            batch_size = o_j.size(0)
            last_o_j = o_j[torch.arange(batch_size), lengths - 1]
            last_t_j = t_j[torch.arange(batch_size), lengths - 1]

            t_til_start = (self.prediction_start - last_t_j) * self.time_scale
            s_t_s = self._s_t(last_o_j, t_til_start)

            preds = np.zeros(batch_size)
            for i in range(batch_size):
                ith_t_s, ith_s_t_s = t_til_start[i], s_t_s[i]
                deltas = torch.arange(0,
                                      (self.integration_end - last_t_j[i]) * self.time_scale,
                                      self.time_scale).to(o_j.device)
                ith_s_deltas = self._s_t(last_o_j[i], deltas[None, :])
                pred_delta = trapz(ith_s_deltas[deltas < ith_t_s].cpu(),
                                   deltas[deltas < ith_t_s].cpu())
                pred_delta += trapz(ith_s_deltas[deltas >= ith_t_s].cpu(),
                                    deltas[deltas >= ith_t_s].cpu()) / ith_s_t_s.item()
                preds[i] = last_t_j[i].cpu().numpy() + pred_delta / self.time_scale

        return preds


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
        print(predicted_events)
        print(gt_events)
        print(event_metric_train)
        print("yo")
        print(predicted_times)
        print(gt_times)
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