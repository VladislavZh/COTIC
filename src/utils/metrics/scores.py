import torch
from sklearn.metrics import (
    r2_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import numpy as np
from typing import List

class R2Score:
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        return float(r2_score(target, pred))

class Precision:
    def __call__(self, pred: torch.Tensor, target: torch.Tensor, labels: List[int]) -> float:
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        return float(precision_score(target, pred, labels, average="weighted"))

class Recall:
    def __call__(self, pred: torch.Tensor, target: torch.Tensor, labels: List[int]) -> float:
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        return float(recall_score(target, pred, labels, average="weighted"))


class F1Score:
    def __call__(self, pred: torch.Tensor, target: torch.Tensor, labels: List[int]) -> float:
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        return float(f1_score(target, pred, labels, average="weighted"))

def MAE(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    return float(np.mean(np.abs(pred - target)))


def RocAuc(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    return float(roc_auc_score(target, pred, multi_class="ovr"))


def Accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = np.argmax(pred.detach().cpu().numpy(), axis=-1)
    target = target.detach().cpu().numpy() - 1
    return accuracy_score(target, pred)
