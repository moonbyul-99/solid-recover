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
        self.logit_scale = nn.Parameter(torch.Tensor([np.log(1 / temperature)]))
    
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
    
class WeightedCLIPLoss(nn.Module):
    """
    Numerically stable weighted CLIP loss for single-cell multi-omics.
    Uses log-sum-exp trick to avoid overflow in exp(logits).
    """
    def __init__(
        self,
        temperature=0.07,
        top_k_ratio=0.1,
        bottom_k_ratio=0.1,
        weight_top=0.1,      
        weight_bottom=2.0    
    ):
        super(WeightedCLIPLoss, self).__init__()
        self.logit_scale = nn.Parameter(torch.tensor([np.log(1.0 / temperature)]))
        self.top_k_ratio = top_k_ratio
        self.bottom_k_ratio = bottom_k_ratio
        self.weight_top = weight_top
        self.weight_bottom = weight_bottom

        # Precompute log weights (constants)
        self.log_weight_top = np.log(weight_top + 1e-8)
        self.log_weight_bottom = np.log(weight_bottom + 1e-8)

    def _compute_weighted_logsumexp(self, logits, W_log):
        """
        Compute log(sum_j exp(logits_ij) * W_ij) = logsumexp(logits_ij + log W_ij)
        Args:
            logits: (N, N)
            W_log: (N, N), log of weights (use -inf for masked entries if needed)
        Returns:
            log_denom: (N,)
        """
        # Add log weights in log-space
        weighted_logits = logits + W_log  # (N, N)
        # Use logsumexp for numerical stability
        log_denom = torch.logsumexp(weighted_logits, dim=1)  # (N,)
        return log_denom
    # def _compute_weight_matrix(self, logits):
    #     """
    #     Compute log-weight matrix W_log based on logits.
    #     For each row i, assign weights to off-diagonal entries based on their rank.
    #     """
    #     device = logits.device
    #     N = logits.size(0)
    #     W_log = torch.zeros(N, N, device=device)

    #     for i in range(N):
    #         off_diag_mask = torch.ones(N, dtype=torch.bool, device=device)
    #         off_diag_mask[i] = False
    #         off_diag_logits = logits[i, off_diag_mask]  # (N-1,)

    #         sorted_vals, sorted_local_idx = torch.sort(off_diag_logits, descending=True)
    #         global_idx = torch.arange(N, device=device)[off_diag_mask][sorted_local_idx]

    #         num_off_diag = N - 1
    #         k_top = max(1, int(self.top_k_ratio * num_off_diag))
    #         k_bottom = max(1, int(self.bottom_k_ratio * num_off_diag))

    #         row_log_weights = torch.zeros(N, device=device)
    #         top_global = global_idx[:k_top]
    #         bottom_global = global_idx[-k_bottom:]

    #         row_log_weights[top_global] = self.log_weight_top
    #         row_log_weights[bottom_global] = self.log_weight_bottom
    #         row_log_weights[i] = 0.0  # diagonal

    #         W_log[i] = row_log_weights

    #     return W_log

    def _compute_weight_matrix(self, logits):
        """
        Vectorized version: compute log-weight matrix W_log for all rows at once.
        """
        device = logits.device
        N = logits.size(0)

        # Create a copy and mask out diagonal (set to -inf so they won't be in top/bottom)
        logits_masked = logits.clone()
        logits_masked.fill_diagonal_(-float('inf'))  # (N, N)

        # Get sorted indices per row (descending)
        # sorted_idx: (N, N-1) but we keep full (N, N) for simplicity; diagonal is last
        _, sorted_idx = torch.sort(logits_masked, descending=True, dim=1)  # (N, N)

        # Number of off-diagonal elements
        num_off_diag = N - 1
        k_top = max(1, int(self.top_k_ratio * num_off_diag))
        k_bottom = max(1, int(self.bottom_k_ratio * num_off_diag))

        # Initialize weight matrix in log space
        W_log = torch.full((N, N), 0.0, device=device)  # default weight = 1 → log(1) = 0

        # For each row i, mark top-k and bottom-k (excluding diagonal)
        # Top-k: first k_top indices in sorted_idx
        top_indices = sorted_idx[:, :k_top]  # (N, k_top)
        # Bottom-k: last k_bottom indices (but skip the very last if it's diagonal? not needed since diag=-inf)
        bottom_indices = sorted_idx[:, -k_bottom:]  # (N, k_bottom)

        # Use advanced indexing to assign weights
        # Create row indices for broadcasting
        row_indices = torch.arange(N, device=device).unsqueeze(1)  # (N, 1)

        # Assign top weights
        W_log[row_indices, top_indices] = self.log_weight_top
        # Assign bottom weights
        W_log[row_indices, bottom_indices] = self.log_weight_bottom

        # Explicitly zero out diagonal (in case bottom includes it, though unlikely due to -inf)
        W_log.fill_diagonal_(0.0)

        return W_log

    def forward(self, rna_emb, atac_emb):
        device = rna_emb.device
        N = rna_emb.size(0)

        rna_emb = F.normalize(rna_emb, dim=-1)
        atac_emb = F.normalize(atac_emb, dim=-1)

        logits = rna_emb @ atac_emb.t()  # (N, N)
        logit_scale = torch.exp(self.logit_scale)
        logits = logit_scale * logits

        # RNA -> ATAC
        W_log_rna2atac = self._compute_weight_matrix(logits)
        log_numer = logits.diag()
        log_denom = self._compute_weighted_logsumexp(logits, W_log_rna2atac)
        loss_rna2atac = -(log_numer - log_denom).mean()

        # ATAC -> RNA: use logits.T as input to weight computation!
        logits_t = logits.t()
        W_log_atac2rna = self._compute_weight_matrix(logits_t)  # ←←← 关键修正！
        log_numer_t = logits_t.diag()  # same as logits.diag()
        log_denom_t = self._compute_weighted_logsumexp(logits_t, W_log_atac2rna)
        loss_atac2rna = -(log_numer_t - log_denom_t).mean()

        return (loss_rna2atac + loss_atac2rna) / 2.0



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
                 temperature: float = 0.07,
                 use_weight = False,
                 top_k_ratio = 0.1,
                 bottom_k_ratio = 0.1,        
                 weight_top = 0.1,
                 weight_bottom = 2.0):
        super().__init__()
        assert cross_recon_1 <= 1.0
        assert cross_recon_2 <= 1.0


        self.vae_beta_1 = vae_beta_1
        self.vae_beta_2 = vae_beta_2

        self.clip_weight = clip_weight
        self.cross_recon_1 = cross_recon_1
        self.cross_recon_2 = cross_recon_2
        
        if not use_weight:
            self.clip_loss = CLIPLoss(temperature)
        else:
            self.clip_loss = WeightedCLIPLoss(temperature, top_k_ratio, bottom_k_ratio, weight_top, weight_bottom)
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
        
        clip_loss = self.clip_loss(x1_dic['z_embed'], x2_dic['z_embed'])  ## original result is computed using z_embed
        #clip_loss = self.clip_loss(x1_dic['z_mu'], x2_dic['z_mu'])  ## clip using z_mu is not good in cross match evaluation 

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


        