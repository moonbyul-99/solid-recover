"""
Define the net structure of solid recover
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math
import numpy as np 
from typing import List, Dict, Union, Any
from sr_loss import *

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
            self.proj = nn.Sequential(nn.Linear(input_dim, output_dim), nn.GELU(), nn.RMSNorm(output_dim), nn.Dropout(dropout_p))
        else:
            self.proj = nn.Sequential(nn.Linear(input_dim, output_dim), nn.GELU(), nn.Dropout(dropout_p))

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
        self.decoder_header = nn.Sequential(nn.Linear(d, feature_num), nn.LeakyReLU())

        fc_blocks = []
        for i in range(1, len(hidden_dims)):
            fc_blocks.append(fc_net(hidden_dims[i-1], hidden_dims[i], use_rmsnorm,use_residual, dropout_p))
        self.fc_blocks = nn.Sequential(*fc_blocks) 
    
    def forward(self,z):
        z = self.fc_blocks(z)
        recon_x = self.decoder_header(z)
        return recon_x

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
        self.mu_proj = nn.Sequential(nn.Linear(d0, embed_dim), nn.RMSNorm(embed_dim))
        self.logvar_proj = nn.Sequential(nn.Linear(d0, embed_dim), nn.RMSNorm(embed_dim))

        self.hidden_dims.append(embed_dim) 
        decoder_hidden_dims = self.hidden_dims.copy()
        decoder_hidden_dims.reverse() 
        self.decoder = feature_decoder(feature_num, decoder_hidden_dims, use_rmsnorm, use_residual, dropout_p)
        self.embed_dim = embed_dim

    def reparam(self, mu, logvar):
        eps = torch.randn_like(mu)
        std = torch.exp(0.5*logvar)
        z = mu + std*eps 
        return z
    
    def get_embedding(self,x):
        z = self.encoder(x)
        z_mu = self.mu_proj(z)
        z_logvar = self.logvar_proj(z)
        z_embed = self.reparam(z_mu, z_logvar)
        return z, z_mu, z_logvar, z_embed

    def forward(self, x):
        # z = self.encoder(x)
        # z_mu = self.mu_proj(z)
        # z_logvar = self.logvar_proj(z)
        # z_embed = self.reparam(z_mu, z_logvar)
        z, z_mu, z_logvar, z_embed = self.get_embedding(x)
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
        self.embed_proj = nn.Sequential(nn.Linear(d0, embed_dim), nn.RMSNorm(embed_dim))
        self.embed_dim = embed_dim

        self.hidden_dims.append(embed_dim)
        decoder_hidden_dims = self.hidden_dims.copy()
        decoder_hidden_dims.reverse() 
        self.decoder = feature_decoder(feature_num, decoder_hidden_dims, use_rmsnorm, use_residual, dropout_p)

    def forward(self, x):
        z = self.encoder(x)
        z_embed = self.embed_proj(z)
        x_recon = self.decoder(z_embed)

        return {'z_encoder': z, 
                'z_embed': z_embed,
                'x_recon': x_recon}

class sr_pair_vae(nn.Module):

    '''
    define the pair vae structure of solid recover 
    '''
    def __init__(self,
                 feature_num_1:int, 
                 feature_num_2:int, 
                 hidden_params_1: Union[Dict[str,int], List[int]],
                 hidden_params_2: Union[Dict[str,int], List[int]],
                 embed_dim: int,
                 use_rmsnorm = True,
                 use_residual = False, 
                 dropout_p = 0.05,
                 clip_temperature = 0.07):  ## TO DO del clip_temperature 
        super().__init__()

        self.model_1 = sr_vae(feature_num=feature_num_1, 
                              hidden_params=hidden_params_1, 
                              embed_dim=embed_dim, 
                              use_rmsnorm=use_rmsnorm, 
                              use_residual=use_residual, 
                              dropout_p=dropout_p)
        self.model_2 = sr_vae(feature_num=feature_num_2, 
                              hidden_params=hidden_params_2, 
                              embed_dim=embed_dim, 
                              use_rmsnorm=use_rmsnorm, 
                              use_residual=use_residual, )
        #self.clip_loss = CLIPLoss(temperature=clip_temperature)
    
    
    def forward(self, x1, x2):
        z, z_mu, z_logvar, z_embed = self.model_1.get_embedding(x1)
        y,y_mu, y_logvar, y_embed = self.model_2.get_embedding(x2)

        #clip_loss = self.clip_loss(z_mu, y_mu)
        x1_z_recon = self.model_1.decoder(z_embed)
        x1_y_recon = self.model_1.decoder(y_embed)

        x2_z_recon = self.model_2.decoder(z_embed)
        x2_y_recon = self.model_2.decoder(y_embed)
        
        x1_dic = {'x_recon': x1_z_recon,
                  'z': z,
                  'z_mu': z_mu,
                  'z_logvar': z_logvar,
                  'z_embed': z_embed}
        x2_dic = {'x_recon': x2_y_recon,
                  'z': y,
                  'z_mu': y_mu,
                  'z_logvar': y_logvar,
                  'z_embed': y_embed}
        
        return {'x1': x1_dic,
                'x2': x2_dic,
                #'clip_loss': clip_loss, 
                'x1_c_recon': x1_y_recon,
                'x2_c_recon': x2_z_recon}

    
