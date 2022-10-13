from pytorch_lightning.callbacks import Callback
import json
import numpy as np
import torch
from typing import Tuple, Union


class RFTraining(Callback):
    def __init__(self):
        self.embeddings = []
        self.return_time_target = []
        self.event_type_target = []
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.embeddings = []
        self.return_time_target = []
        self.event_type_target = []
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        mask = batch[1].ne(0)[:,1:]
        event_time = batch[0]
        return_time = event_time[:,1:] - event_time[:,:-1]
        event_type = batch[1][:,1:]
        
        self.embeddings.append(outputs["out"][0][:,1:-1,:][mask,:].detach().cpu().numpy())
        self.return_time_target.append(return_time[mask].detach().cpu().numpy())
        self.event_type_target.append(event_type[mask].detach().cpu().numpy())
    
    def on_train_epoch_end(self, trainer, pl_module):
        X = np.concatenate(self.embeddings, axis = 0)
        y_reg = np.concatenate(self.return_time_target, axis = 0)
        y_cls = np.concatenate(self.event_type_target, axis = 0)
        
        pl_module.net.regressor.fit(X, y_reg)
        pl_module.net.classifier.fit(X, y_cls)
        