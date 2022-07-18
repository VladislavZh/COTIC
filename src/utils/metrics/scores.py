import torch
from sklearn.metrics import r2_score, roc_auc_score

class R2Score:
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        return float(r2_score(target, pred))
    
class RocAuc:
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        return float(roc_auc_score(target, pred))
