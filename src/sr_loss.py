'''
this file contains the loss function used in solid recover
'''

import torch  
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np

class CLIPLoss(nn.Module):
    """
    Implementation of the CLIP loss function.
    """
    
    def __init__(self, temperature=0.07):
        super(CLIPLoss, self).__init__()
        self.logit_scale = nn.Parameter(torch.Tensor([np.log(1 / 0.07)]))
    
    def reset_temperature(self, temperature = None):
        if temperature is not None:
            self.temperature = temperature

    def forward(self, cell_1, cell_2):
        cell_1 = F.normalize(cell_1, dim=-1)
        cell_2 = F.normalize(cell_2, dim=-1)

        logits = cell_1 @ cell_2.t()
        logits_cell_1 = torch.exp(self.logit_scale) * logits
        logits_cell_2 = logits_cell_1.t()
        
        ground_truth = torch.arange(len(cell_1), dtype=torch.long, device=cell_1.device)
        loss_12 = F.cross_entropy(logits_cell_1, ground_truth)
        loss_21 = F.cross_entropy(logits_cell_2, ground_truth)

        return (loss_12 + loss_21) / 2 




class VAE_loss(nn.Module):
    def __init__(self, kl_weight = 1.0):
        super().__init__() 
        self.kl_weight = kl_weight
    @staticmethod
    def _kl_loss(mu, logvar):
        persample_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis = 1)
        return persample_kl.mean()
    def forward(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='none')
        recon_loss = recon_loss.sum(axis = 1).mean() 

        kl_loss = self._kl_loss(mu, logvar)

        loss = recon_loss + self.kl_weight * kl_loss 
        return {'loss': loss, 
                'kl_loss': kl_loss,
                'recon_loss': recon_loss}

class AE_loss(nn.Module):
    def __init__(self, kl_weight =None):
        super().__init__() 
        self.kl_weight = kl_weight
    def forward(self, recon_x, x):
        recon_loss = F.mse_loss(recon_x, x, reduction='none')
        recon_loss = recon_loss.sum(axis = 1).mean() 

        return {'loss': recon_loss, 
                'recon_loss': recon_loss}
    


        