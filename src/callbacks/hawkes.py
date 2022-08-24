import torch


class Hawkes:
    def __init__(self, baselines, adjacencies, decays):
        self.baselines = baselines
        self.adjacencies = adjacencies
        self.decays = decays

    def intensity(self, t, event_times, event_types):
        """
        Returns intensity function in the momemt of time t given the sequence
        If in_point is true change the inequality from t_i < t to t_i <= t, this will be useful for sequence generation
        """
        if len(event_time) > 0:
            delta_t = t - event_time
            event_types = event_types[delta_t > 0]
            delta_t = delta_t[delta_t > 0]
            
            event_types_ohe = torch.zeros(event_types.shape[0], len(self.baselines))
            for i in range(event_types.shape[0]):
                event_types_ohe[i,event_types[i]] = 1 
      
            lambdas = self.baselines + torch.sum((self.adjacencies.unsqueeze(0) * \
                                                  self.decays.unsqueeze(0) * \
                                                  torch.exp(-self.decays.unsqueeze(0) \
                                                          * delta_t[:,None,None])) *\
                                                  event_types_ohe.unsqueeze(1), dim = (0,2))
            return lambdas
        else:
            return self.baselines