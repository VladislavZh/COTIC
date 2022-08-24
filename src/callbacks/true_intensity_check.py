from pytorch_lightning.callbacks import Callback
import json
import numpy as np
import torch
from typing import Tuple
import os

from .hawkes import Hawkes

import matplotlib.pyplot as plt
from comet_ml import Experiment


class PrintingCallback(Callback):
    def __init__(
        self,
        data_path: str,
        sim_size = 40
    ) -> None:
        self.data_path = data_path
        files = os.listdir(self.data_path)
        if 'process_params.json' in files:
            with open(data_path+'/process_params.json', 'r') as f:
                params = json.load(f)
            self.true_model = Hawkes(torch.Tensor(params['baseline']), torch.Tensor(params['adjacency']), torch.Tensor(params['decay']))
        else:
            self.true_model = None
        
        self.sim_size = sim_size
        
        self.experiment = Experiment(
            api_key="eTfGsxdNWCvov8CuHEC0ug9ko",
            project_name="sinth-exps-imgs",
            workspace="vladislavzh",
        )
            
    @staticmethod
    def __add_sim_times(
        times: torch.Tensor,
        sim_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes batch of times and events and adds sim_size auxiliar times
        
        args:
            times  - torch.Tensor of shape (bs, max_len), that represents event arrival time since start
            featuers - torch.Tensor of shape (bs, max_len, d), that represents event features
            sim_size - int, number of points to simulate
            
        returns:
            bos_full_times  - torch.Tensor of shape(bs, (sim_size + 1) * (max_len - 1) + 1) that consists of times and sim_size auxiliar times between events
            bos_full_features - torch.Tensor of shape(bs, (sim_size + 1) * (max_len - 1) + 1, d) that consists of event features and sim_size zeros between events
        """
        delta_times = times[:,1:] - times[:,:-1]
        sim_delta_times = (torch.rand(list(delta_times.shape)+[sim_size]).to(times.device) * delta_times.unsqueeze(2)).sort(dim=2).values
        full_times = torch.concat([sim_delta_times.to(times.device),delta_times.unsqueeze(2)], dim = 2)
        full_times = full_times + times[:,:-1].unsqueeze(2)
        full_times[delta_times<0,:] = 0
        full_times = full_times.flatten(1)
        bos_full_times = torch.concat([torch.zeros(times.shape[0],1).to(times.device), full_times], dim = 1)
        
        return bos_full_times
    
    def on_validation_epoch_end(self, trainer, pl_module):
        times, events = trainer.datamodule.data_val[0]
        non_pad_mask = events.ne(0)
        times = times[non_pad_mask]
        events = events[non_pad_mask]
        
        times = times.unsqueeze(0)
        events = events.unsqueeze(0)
        
        full_times = self.__add_sim_times(times, self.sim_size)
        
        if self.true_model:
            tracked_intensity_true = []
            for t in full_times[0]:
                tracked_intensity_true.append(float(torch.sum(self.true_model.intensity(t.cpu(), times[0].cpu(), events[0].cpu()))))
            tracked_intensity_true = np.array(tracked_intensity_true)
        
        enc_output = pl_module(times, events)
        event_time = torch.concat([torch.zeros(times.shape[0],1).to(times.device), times], dim = 1)
        non_pad_mask = torch.concat([torch.ones(event_time.shape[0],1).to(event_time.device),  events.ne(0).long()], dim = 1).long()
        tracked_intensity = torch.sum(model.final(full_times, event_time, enc_output, non_pad_mask, self.sim_size), dim=-1).detach().cpu().numpy() # shape = (bs, (num_samples + 1) * L + 1, num_types)
        
        fig = plt.figure(figsize=(16,9), dpi=300)
        if self.true_model:
            plt.plot(full_times.detach().cpu().numpy(), tracked_intensity_true, label = "True model")
        plt.plot(full_times.detach().cpu().numpy(), tracked_intensity, label = "Predicted model")
        plt.legend(loc="upper right", bbox_to_anchor=(1,1))
        self.experiment.log_figure(figure=plt)
        