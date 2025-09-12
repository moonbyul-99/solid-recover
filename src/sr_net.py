"""
Define the net structure of solid recover
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math
import numpy as np 
from typing import List, Dict, Union, Any

class fc_net(nn.Module):

    '''
    A linear block

    input_dim: int, input dimension
    output_dim: int, output dimension
    use_rmsnorm: bool, whether to use rmsnorm
    use_residual: bool, whether to use residual connection
    dropout_p: float, dropout probability
    '''
    def __init__(self, input_dim :int, output_dim:int, use_rmsnorm = True, use_residual = False, dropout_p = 0.05):
        super(fc_net, self).__init__()

        if input_dim != output_dim: 
            self.use_residual = False
        else: 
            self.use_residual = use_residual 

        self.use_rmsnorm = use_rmsnorm 

        if self.use_rmsnorm:
            self.proj = nn.Sequential([nn.Linear(input_dim, output_dim), nn.GELU(), nn.RMSNorm(output_dim), nn.Dropout(dropout_p)])
        else:
            self.proj = nn.Sequential([nn.Linear(input_dim, output_dim), nn.GELU(), nn.Dropout(dropout_p)])

    def forward(self, x):
        if self.use_residual:
            return x + self.proj(x) 
        return self.proj(x)

class feature_encoder(nn.Module):
    '''
    Feature encoder for sr model 

    feature num: int (number of features)
    hidden_params: Union[Dict[str,int], List[int]] if list, the hidden params records the number of hidden units in each layer, if dict, 
                    must contain keys hidden_dim, block_num, then have block_num hidden layers with hidden_dim units
    use_rmsnorm: bool
    use_residual: bool
    dropout: float
    '''
    def __init__(self, feature_num: int, 
                 hidden_params: Union[Dict[str,int], List[int]],
                 use_rmsnorm = True,
                 use_residual = False, 
                 dropout_p = 0.05): 

        super().__init__() 

        self.hidden_params = hidden_params
        ## parse hidden_params 

        if isinstance(hidden_params, dict):
            if not {'hidden_dim', 'block_num'}.issubset(hidden_params.keys()):
                raise ValueError('hidden_params is dict, must contain "hidden_dim" and "block_num"')
            hidden_dim = hidden_params['hidden_dim']
            block_num = hidden_params['block_num']
            hidden_dims = [hidden_dim]*block_num 
        elif isinstance(hidden_params, list):
            hidden_dims = hidden_params
        else: 
            raise TypeError("hidden_params must be either a dict (with 'hidden_dim', 'block_num') or a list of ints")
        
        self.hidden_dims = hidden_dims
        d = hidden_dims[0]
        self.encoder_header = fc_net(input_dim = feature_num, output_dim = d, use_rmsnorm = use_rmsnorm,use_residual = False, dropout_p = dropout_p)

        fc_blocks = []
        for i in range(1, len(hidden_dims)):
            fc_blocks.append(fc_net(hidden_dims[i-1], hidden_dims[i], use_rmsnorm,use_residual, dropout_p))
        self.fc_blocks = nn.Sequential(*fc_blocks) 
    
    def forward(self,x):
        z = self.encoder_header(x)
        z = self.fc_blocks(z)
        return z 

class feature_decoder(nn.Module):
    '''
    Feature decoder for sr model 

    feature num: int (number of features)
    hidden_params: Union[Dict[str,int], List[int]] if list, the hidden params records the number of hidden units in each layer, if dict, 
                    must contain keys hidden_dim, block_num, then have block_num hidden layers with hidden_dim units
    use_rmsnorm: bool
    use_residual: bool
    dropout: float
    '''
    def __init__(self, feature_num: int, 
                 hidden_params: Union[Dict[str,int], List[int]],
                 use_rmsnorm = True,
                 use_residual = False, 
                 dropout_p = 0.05): 

        super().__init__() 

        self.hidden_params = hidden_params
        ## parse hidden_params 

        if isinstance(hidden_params, dict):
            if not {'hidden_dim', 'block_num'}.issubset(hidden_params.keys()):
                raise ValueError('hidden_params is dict, must contain "hidden_dim" and "block_num"')
            hidden_dim = hidden_params['hidden_dim']
            block_num = hidden_params['block_num']
            hidden_dims = [hidden_dim]*block_num 
        elif isinstance(hidden_params, list):
            hidden_dims = hidden_params
        else: 
            raise TypeError("hidden_params must be either a dict (with 'hidden_dim', 'block_num') or a list of ints")
        self.hidden_dims = hidden_dims

        d = hidden_dims[-1]
        self.decoder_header = nn.Sequential([nn.Linear(d, feature_num), nn.LeakyReLU()])

        fc_blocks = []
        for i in range(1, len(hidden_dims)):
            fc_blocks.append(fc_net(hidden_dims[i-1], hidden_dims[i], use_rmsnorm,use_residual, dropout_p))
        self.fc_blocks = nn.Sequential(*fc_blocks) 
    
    def forward(self,z):
        z = self.fc_blocks(z)
        z = self.decoder_header(x)
        return z 

class sr_vae(nn.Module):
    """
    Define the vae structure of solid recover
    """
    def __init__(self, 
                 feature_num: int, 
                 hidden_params: Union[Dict[str,int], List[int]],
                 embed_dim: int,
                 use_rmsnorm = True,
                 use_residual = False, 
                 dropout_p = 0.05):
        super().__init__()

        self.encoder = feature_encoder(feature_num, hidden_params, use_rmsnorm, use_residual, dropout_p)

        self.hidden_dims = self.encoder.hidden_dims

        d0 = self.hidden_dims[-1]
        self.mu_proj = nn.Sequential([nn.Linear(d0, embed_dim), nn.RMSNorm(embed_dim)])
        self.logvar_proj = nn.Sequential([nn.Linear(d0, embed_dim), nn.RMSNorm(embed_dim)])

        decoder_hidden_dims = self.hidden_dims.append(embed_dim) 
        decoder_hidden_dims.reverse() 
        self.decoder = feature_decoder(feature_num, decoder_hidden_dims, use_rmsnorm, use_residual, dropout_p)

    def reparam(self, mu, logvar):
        eps = torch.randn_like(mu)
        std = torch.exp(0.5*logvar)
        z = mu + std*eps 
        return z

    def forward(self, x):
        z = self.encoder(x)
        z_mu = self.mu_proj(z)
        z_logvar = self.logvar_proj(z)
        z_embed = self.reparam(z_mu, z_logvar)
        x_recon = self.decoder(z_embed)

        return {'z_encoder': z, 
                'z_mu': z_mu,
                'z_logvar': z_logvar,
                'z_embed': z_embed,
                'x_recon': x_recon}

class sr_ae(nn.Module):
    """
    Define the ae structure of solid recover
    """
    def __init__(self, 
                 feature_num: int, 
                 hidden_params: Union[Dict[str,int], List[int]],
                 embed_dim: int,
                 use_rmsnorm = True,
                 use_residual = False, 
                 dropout_p = 0.05):
        super().__init__()

        self.encoder = feature_encoder(feature_num, hidden_params, use_rmsnorm, use_residual, dropout_p)

        self.hidden_dims = self.encoder.hidden_dims

        d0 = self.hidden_dims[-1]
        self.embed_proj = nn.Sequential([nn.Linear(d0, embed_dim), nn.RMSNorm(embed_dim)])

        decoder_hidden_dims = self.hidden_dims.append(embed_dim) 
        decoder_hidden_dims.reverse() 
        self.decoder = feature_decoder(feature_num, decoder_hidden_dims, use_rmsnorm, use_residual, dropout_p)

    def forward(self, x):
        z = self.encoder(x)
        z_embed = self.embed_proj(z)
        x_recon = self.decoder(z_embed)

        return {'z_encoder': z, 
                'z_embed': z_embed,
                'x_recon': x_recon}





    
# class reparam_module(nn.Module): 
#     """
#     VAE model reparam steps
#     """

#     def __init__(self,
#                  input_dim,
#                  output_dim):
#         super().__init__() 
#         self.mean_net = nn.Linear(input_dim, output_dim)
#         self.log_var_net = nn.Linear(input_dim, output_dim) 
#     def kld_loss(self, log_var, mean):

#         #  calculate kld loss for each sample (B,d) --> (B)
#         per_sample_kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
#         #  get mean kld loss for the batch (B) --> 1
#         return torch.mean(per_sample_kld)
#         #return  #-0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) # wrong code, no batch mean
#     def forward(self, x):
#         mean = self.mean_net(x)

#         log_var = self.log_var_net(x) 
#         std = torch.exp(0.5 * log_var)

#         eps = torch.randn_like(mean)
#         z = mean + eps * std
        
#         loss = self.kld_loss(log_var, mean)
#         return {'cell_embed': z,
#                 'mean': mean,
#                 'log_var': log_var,
#                 'kld_loss': loss}

# class classification_head(nn.Module):
#     '''
#     2 layer MLP used for classification
#     '''

#     def __init__(self,
#                  input_dim,
#                  output_dim):
#         super().__init__()
#         self.projection_1 = nn.Linear(input_dim,
#                                     output_dim)
#         self.projection_2 = nn.Linear(output_dim,
#                                     output_dim)
#         self.classification_criterion = nn.CrossEntropyLoss()

#         self.activation = nn.GELU()
#     def forward(self, x, label):
#         logits = self.projection_1(x)
#         logits = self.activation(logits)
#         logits = self.projection_2(logits)
#         loss = self.classification_criterion(logits, label)
#         return {'ce_loss': loss,
#                 'logits': logits}


# class sr_single_omic(nn.Module):
#     '''
#     single omic model
#     support both AE and VAE model
#     support training with or without labels,
#     use symmetrical encoder and decoder, i.e. d, d1, d2, ..., d[-1], ... , d2, d1, d
#     '''

#     def __init__(self, feature_num, hidden_dims, dropout, layernorm_eps, activation, input_dropout, vae_weight, ce_weights, class_dict):
#         """
#         Args:

#         feature_num: int, the number of genes
#         hidden_dims: list of int, the number of hidden dim, feature_num --> h[0] --> h[1] --> ... --> h[-1]
#         dropout: dropout rate, float
#         activation: key of activateion, str in ('relu', 'rrelu','sigmoid','gelu','leaky_relu','tanh')
#         layernorm_eps: float, the eps of layer norm
#         input_dropout: float,  dropout rate of the input data
#         vae_based: bool, whether use vae based model, default False
#         ce_loss: bool, whether use classification loss, default False
#         vae_weight: float
#         ce_weights: dict of float, key is class name, value is the subloss weight,loss = recon_loss + vae_weight * kld_loss + \sum_{i} ce_weights[i]*ce_losses[i]
#         class_dict: dict, key is class name, value is the class num
#         """
        
#         super().__init__()
#         self.encoder_net = feature_encoder(feature_num= feature_num,
#                                            hidden_dims= hidden_dims,
#                                            dropout= dropout,
#                                            layernorm_eps= layernorm_eps,
#                                            activation= activation)
        
#         reverse_dims = []
#         for i,_ in enumerate(hidden_dims):
#             reverse_dims.append(hidden_dims[len(hidden_dims)-1-i])
#         self.decoder_net = feature_decoder(feature_num= feature_num,
#                                            hidden_dims= reverse_dims,
#                                            dropout= dropout,
#                                            activation= activation)
        
#         self.mask = nn.Dropout(p = input_dropout)

#         if vae_weight > 0:
#             self.reparam_module = reparam_module(input_dim = hidden_dims[-1],
#                                                  output_dim = hidden_dims[-1])
#             self.model_state = 'vae'
#         else:
#             self.reparam_module = None 
#             self.model_state = 'ae'

#         if ce_weights is not None and any(ce_weights.values()):
#             self.classification_heads = nn.ModuleDict() 
#             for key in class_dict:
#                 self.classification_heads[key] = classification_head(input_dim = hidden_dims[-1],
#                                                                      output_dim = class_dict[key])
#         else:
#             self.classification_heads = None
        
#         self.input_dropout = input_dropout
#         self.feature_num = feature_num
#         self.hidden_dims = hidden_dims
#         self.vae_weight = vae_weight 
#         self.ce_weights = ce_weights 
#         self.class_dict = class_dict
        
#     def recon_loss(self, x, x_recon, mean = 'none'):
#         '''
#         loss scale balance:
#         if 'none': the recon loss of 30k gene is about 2k
#         if 'mean': the recon loss of 30k gene is about 0.1 

#         during pretraining, use 'none', mainly focus on reconstruction error 
#         during alignment, use 'mean', balance the recon error and clip loss
#         '''

#         if mean == 'none':
#             error = F.mse_loss(x, x_recon, reduction='none')
#             error = error.sum(axis = 1).mean()
#         if mean == 'mean':
#             error = F.mse_loss(x, x_recon, reduction='mean')
#         return error
    
#     def encode_forward(self, x):
#         '''
#         encode for both ae and vae
#         '''
#         mask_x = self.mask(x)
#         cell_embed = self.encoder_net(mask_x)
        
#         if self.model_state == 'vae':
#             vae_output = self.reparam_module(cell_embed)
#             cell_embed = vae_output['cell_embed']
#             kld_loss = vae_output['kld_loss']
#             output = {'cell_embed': cell_embed,
#                   'kld_loss': kld_loss}
        
#         else:     
#             output = {'cell_embed': cell_embed}
#         return output
#     def forward(self, feature, label_dic):

#         encode_out = self.encode_forward(feature) 
#         cell_embed = encode_out['cell_embed']

#         x_recon = self.decoder_net(cell_embed)
#         recon_loss = self.recon_loss(feature, x_recon)

#         if self.model_state  == 'ae':
#             output = {'cell_embed': cell_embed,
#                       'recon_loss': recon_loss}
#             kld_loss = 0
#         else:
#             kld_loss = encode_out['kld_loss']
#             output = {'cell_embed': cell_embed,
#                       'recon_loss': recon_loss,
#                       'kld_loss': kld_loss}

#         if label_dic != None:
#             ## calculate ce_loss
#             ce_loss = 0 
#             for key in self.classification_heads:
#                 head_output = self.classification_heads[key](cell_embed, label_dic[key])
#                 output[f'{key}_celoss'] = head_output['ce_loss']
#                 ce_loss += self.ce_weights[key] * head_output['ce_loss']
            
#             output['ce_loss'] = ce_loss
#         else: 
#             ce_loss = 0
        
#         ## calculate total loss 
#         loss = recon_loss + ce_loss + self.vae_weight * kld_loss
#         output['loss'] = loss 
#         return output
    
# class CLIPLoss(nn.Module):
#     """
#     Implementation of the CLIP loss function.
#     """
    
#     def __init__(self, temperature=0.1):
#         super(CLIPLoss, self).__init__()
#         self.temperature = temperature
    
#     def reset_temperature(self, temperature = None):
#         if temperature is not None:
#             self.temperature = temperature

#     def forward(self, cell_1, cell_2):
#         cell_1 = F.normalize(cell_1, dim=-1)
#         cell_2 = F.normalize(cell_2, dim=-1)

#         logits_cell_1 = torch.div(
#             torch.matmul(cell_1, cell_2.t()),
#             self.temperature
#         )
#         logits_cell_2 = logits_cell_1.t()
        
#         ground_truth = torch.arange(len(cell_1), dtype=torch.long, device=cell_1.device)
#         loss_12 = F.cross_entropy(logits_cell_1, ground_truth)
#         loss_21 = F.cross_entropy(logits_cell_2, ground_truth)

#         return (loss_12 + loss_21) / 2 
    
# ############################################################################################
# class multi_model(nn.Module):

#     """
#     multi model for paired omics data, 
#     support alignment two pretrained single omic data,
#     support training and align a multi omics model from scratch
#     """

#     """
#     TO DO: support model define from config
#     """
#     def __init__(self,
#                  rna_model,
#                  ga_model,
#                  temperature=0.1,
#                  tau = 0.5,
#                  rna_weight = 1,
#                  ga_weight = 1):
#         super().__init__()
    
#         '''
#         rna_model: a pretrained rna model 
#         ga_model: a pretrained ga model 
#         temperature: clip temperature
#         tau: clip loss weight
#         '''
#         self.rna_model = rna_model
#         self.ga_model = ga_model

#         self.rna_model.mask = nn.Dropout(0)
#         self.ga_model.mask = nn.Dropout(0)

#         self.cliploss = CLIPLoss(temperature)
#         self.tau = tau
#         self.rna_weight = rna_weight
#         self.ga_weight = ga_weight
#     def freeze_param(self):
#         for param in self.rna_model.encoder_net.encoder_header.parameters():
#             param.requires_grad = False
#         for param in self.rna_model.decoder_net.decoder_header.parameters():
#             param.requires_grad = False
            
#         for param in self.ga_model.encoder_net.encoder_header.parameters():
#             param.requires_grad = False
#         for param in self.ga_model.decoder_net.decoder_header.parameters():
#             param.requires_grad = False
    
#     def rna2ga(self, rna):
#         '''
#         rna -- rna_embed --(ga decoder)-- ga_recon
#         '''
#         output = self.rna_model.encode_forward(rna)
#         rna_embed = output['cell_embed']

#         ga_recon = self.ga_model.decoder_net(rna_embed)
#         return {'rna_embed': rna_embed,
#                 'ga_pred': ga_recon}
    
#     def ga2rna(self, ga):
#         '''
#         ga -- ga_embed --(rna decoder)-- rna_recon
#         '''
#         output = self.ga_model.encode_forward(ga)
#         ga_embed = output['cell_embed']

#         rna_recon = self.rna_model.decoder_net(ga_embed)
#         return {'ga_embed': ga_embed,
#                 'rna_pred': rna_recon}
    
#     def multi2embed(self, rna, ga):
#         rna_out = self.rna_model.encode_forward(rna)
#         ga_out = self.ga_model.encode_forward(ga)
#         return {'ga_embed': ga_out['cell_embed'],
#                 'rna_embed': rna_out['cell_embed']}

#     def forward(self, rna, ga):

#         rna_output = self.rna_model.encode_forward(rna)
#         ga_output = self.ga_model.encode_forward(ga)

#         if 'kld_loss' in rna_output:
#             rna_kld = rna_output['kld_loss']
#         else:
#             rna_kld = 0
        
#         if 'kld_loss' in ga_output:
#             ga_kld = ga_output['kld_loss']
#         else:
#             ga_kld = 0

#         rna_embed = rna_output['cell_embed']
#         ga_embed = ga_output['cell_embed']

#         ## cross prediction
#         rna_recover = self.rna_model.decoder_net(ga_embed)
#         ga_recover = self.ga_model.decoder_net(rna_embed)

#         #rna_loss = self.rna_model.recon_loss(rna, rna_recover, mean = 'none') + self.rna_model.vae_weight * rna_kld
#         #ga_loss = self.ga_model.recon_loss(ga, ga_recover, mean = 'none') + self.ga_model.vae_weight * ga_kld
#         rna_loss = self.rna_model.recon_loss(rna, rna_recover, mean = 'mean') #+ self.rna_model.vae_weight * rna_kld
#         ga_loss = self.ga_model.recon_loss(ga, ga_recover, mean = 'mean') #+ self.ga_model.vae_weight * ga_kld

#         ## clip loss

#         clip_loss = self.cliploss(rna_embed, ga_embed)
#         loss = self.rna_weight * rna_loss + self.ga_weight * ga_loss + self.tau * clip_loss 
        
#         output = {'loss': loss,
#                   'clip_loss': clip_loss,
#                   'ga_loss': ga_loss,
#                   'rna_loss': rna_loss,
#                   'rna_embed': rna_embed,
#                   'ga_embed': ga_embed,
#                   'rna_kld_loss': rna_kld,
#                   'ga_kld_loss': ga_kld}
#         return output
