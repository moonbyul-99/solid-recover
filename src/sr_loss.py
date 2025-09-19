'''
this file contains the loss function used in solid recover
'''

import torch  
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np
from typing import List, Dict, Union, Any

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

class recon_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, recon_x, x):

        recon_loss = F.mse_loss(recon_x, x, reduction='none')
        recon_loss = recon_loss.sum(axis = 1).mean()

        return recon_loss 


class VAE_loss(nn.Module):
    def __init__(self, kl_weight = 1.0):
        super().__init__() 
        self.kl_weight = kl_weight
        self.recon_loss = recon_Loss()
    @staticmethod
    def _kl_loss(mu, logvar):
        persample_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis = 1)
        return persample_kl.mean()
    def forward(self, recon_x, x, mu, logvar):
        recon_loss = self.recon_loss(recon_x, x)

        kl_loss = self._kl_loss(mu, logvar)

        loss = recon_loss + self.kl_weight * kl_loss 
        return {'loss': loss, 
                'kl_loss': kl_loss,
                'recon_loss': recon_loss}

class AE_loss(nn.Module):
    def __init__(self, kl_weight =None):
        super().__init__() 
        self.kl_weight = kl_weight
        self.recon_loss = recon_Loss()
    def forward(self, recon_x, x):
        recon_loss = self.recon_loss(recon_x, x)

        return {'loss': recon_loss, 
                'recon_loss': recon_loss}
    
class VAE_clip_loss(nn.Module):

    '''
    Loss = recon_1 * (recon_loss_1 + vae_beta_1 * kl_loss_1) + recon_2 * (recon_loss_2 + vae_beta_2 * kl_loss_2) + clip_weight * clip_loss + 
            (1-recon_1) * cross_recon_loss_1 + (1-recon_2) * cross_recon_loss_2
    '''
    def __init__(self,                  
                 vae_beta_1: float = 1.0, 
                 vae_beta_2: float = 1.0, 
                 clip_weight: float = 1.0,
                 cross_recon_1: float = 0.2,
                 cross_recon_2: float = 0.2,
                 temperature: float = 0.07,):
        super().__init__()
        assert cross_recon_1 <= 1.0
        assert cross_recon_2 <= 1.0


        self.vae_beta_1 = vae_beta_1
        self.vae_beta_2 = vae_beta_2

        self.clip_weight = clip_weight
        self.cross_recon_1 = cross_recon_1
        self.cross_recon_2 = cross_recon_2
        
        self.clip_loss = CLIPLoss(temperature)
        self.vae_loss_1 = VAE_loss(self.vae_beta_1)
        self.vae_loss_2 = VAE_loss(self.vae_beta_2)
        self.recon_loss = recon_Loss()
    
    def forward(self, x1, x2, sr_pair_out:Dict):

        ''' calculate the vae loss for omic 1'''
        x1_dic = sr_pair_out['x1']
        x2_dic = sr_pair_out['x2']

        vae_loss_1 = self.vae_loss_1(recon_x = x1_dic['x_recon'],
                                     x = x1, 
                                     mu = x1_dic['z_mu'],
                                     logvar = x1_dic['z_logvar'])
        
        vae_loss_2 = self.vae_loss_2(recon_x = x2_dic['x_recon'],
                                     x = x2, 
                                     mu = x2_dic['z_mu'],
                                     logvar = x2_dic['z_logvar'])
        
        clip_loss = self.clip_loss(x1_dic['z_embed'], x2_dic['z_embed'])

        cross_loss_1 = self.recon_loss(sr_pair_out['x1_c_recon'], x1)
        cross_loss_2 = self.recon_loss(sr_pair_out['x2_c_recon'], x2)

        loss = self.cross_recon_1 * cross_loss_1 + (1-self.cross_recon_1) * vae_loss_1['recon_loss'] + \
                self.cross_recon_2 * cross_loss_2 + (1-self.cross_recon_2) * vae_loss_2['recon_loss'] + \
                + self.vae_beta_1 * vae_loss_1['kl_loss'] + self.vae_beta_2 * vae_loss_2['kl_loss'] + \
                  self.clip_weight * clip_loss
        
        return {'loss': loss, 
                'recon_loss_1': vae_loss_1['recon_loss'],
                'recon_loss_2': vae_loss_2['recon_loss'],
                'cross_loss_1': cross_loss_1,
                'cross_loss_2': cross_loss_2,
                'kl_loss_1': vae_loss_1['kl_loss'],
                'kl_loss_2': vae_loss_2['kl_loss'],
                'clip_loss': clip_loss,}


def batch_kl_(mu1, logvar1, mu2, logvar2):
    """
    compute the KL divergence between two batches of Gaussian distributions.

    Args:
        mu1:      [B,D]
        logvar1:  [B, D] 
        mu2:      [B, D] 
        logvar2:  [B, D] 

    return :
        kl_matrix: [B, B] —— kl_matrix[i, j] = KL(N(mu1[i], Sigma1[i]) || N(mu2[j], Sigma2[j]))
    """
    B, D = mu1.shape

    # broadcast mu and logver 
    mu1_expanded = mu1.unsqueeze(1)                    # [B, 1, D]
    logvar1_expanded = logvar1.unsqueeze(1)            # [B, 1, D]

    mu2_expanded = mu2.unsqueeze(0)                    # [1, B, D]
    logvar2_expanded = logvar2.unsqueeze(0)            # [1, B, D]

    var1 = logvar1_expanded.exp()  + 1e-8                    # diag(Sigma1), [B, 1, D]
    var2 = logvar2_expanded.exp()  + 1e-8                    # diag(Sigma2), [1, B, D]

    mu_diff = mu1_expanded - mu2_expanded             # [B, B, D]
    # 1. trace(Sigma2^{-1} Sigma1) = sum_d (Sigma1_dd / Sigma2_dd)
    trace_term = (var1 / var2).sum(dim=-1)  # [B, B]

    # 2. (mu1 - mu2)^T Sigma2^{-1} (mu1 - mu2) = sum_d (diff_d^2 / var2_d)
    mahalanobis_term = (mu_diff ** 2 / var2).sum(dim=-1)  # [B, B]

    # 3. dimension term
    D_tensor = torch.tensor(D, dtype=torch.float32, device=mu1.device)

    # 4. log(det Sigma2 / det Sigma1) = log(prod Sigma2_dd) - log(prod Sigma1_dd)
    #                                 = sum(log Sigma2_dd) - sum(log Sigma1_dd)
    # 注意：log(var) = log(diag(Sigma))，所以 sum(log(var)) = log(det(Sigma))
    logdet_ratio_term = (logvar2_expanded - logvar1_expanded).sum(dim=-1)  # [B, B]

    # merge all terms
    kl_matrix = 0.5 * (trace_term + mahalanobis_term - D_tensor + logdet_ratio_term)

    return kl_matrix  # [B, B]

# class CLIP_VAE(nn.Module):


        