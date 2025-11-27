import numpy as np 
import scanpy as sc 
import muon as mu
import pandas as pd
from sr_model import *
from sr_net import *
from anndata import AnnData
from scipy import sparse
import torch
from torch.utils.data import Dataset, DataLoader

class infer_dataset(Dataset):
    def __init__(self, X):
        self.X = X
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        x1 = self.X[idx,:]
        return x1



class sr_grn:

    def __init__(self, 
                 rna_model, 
                 atac_model,
                 device = 'cuda'):
        self.rna_model = rna_model.to(device)
        self.atac_model = atac_model.to(device)
        self.device = device


        '''set loss just for inference, do not influence the result'''
        self.rna_model.set_loss(1.0)
        self.atac_model.set_loss(1.0)

    def load_data(self, rna_data: AnnData, atac_data: AnnData, cluster_label: str):
        print(f' Please make sure the features of input data are the same as the model input layer')
        print(f' Please make sure the cluster label in both rna_data and atac_data obs')
        print(f' Please make sure the rna data and atac data has the same index')
        self.rna = rna_data
        self.atac = atac_data 
        self.cluster_label = cluster_label
        self.cluster_info = self.rna.obs.loc[:,cluster_label].values

    def _sparse_array(self, X):
        if sparse.issparse(X):
            X = X.toarray() 
            return X
        
    def _rna2atac(self,X):
        self.rna_model.eval()
        self.atac_model.eval()

        infer_dataset = infer_dataset(X)
        infer_loader = DataLoader(infer_dataset, batch_size = 128, shuffle = False)

        embed_list = []
        recon_list = []
        with torch.no_grad():
            for batch in infer_loader:
                batch = batch.to(self.device)
                z, z_mu, z_logvar, z_embed = self.rna_model.get_embedding(batch)
                atac_recon = self.atac_model.decoder(z_embed)

                embed_list.append(z_mu.detach().cpu().numpy())
                recon_list.append(atac_recon.detach().cpu().numpy())
        embed = np.concatenate(embed_list)1
        recon = np.concatenate(recon_list)
        return {'embed': embed, 'recon': recon}

    def tf_re_perturb(self, tf_name:str, re_name:str, cluster_id: str, batch_size: int = 256, down_rate: float = 0.0, up_rate: float = 2.0):
        if tf_name not in self.rna.var.index:
            print(f'{tf_name} not in rna data, break')
            return None 
        if re_name not in self.atac.var.index:
            print(f'{re_name} not in atac data, break')
            return None
        if cluster_id not in self.cluster_info:
            print(f'{cluster_id} not in cluster info, break')
            return None 

        tf_id = self.rna.var.index.get_loc(tf_name)

        n = self.rna.shape[0]
        sample_idx = self.cluster_info == cluster_id
        sample_idx = np.arange(n)[sample_idx]
        sample_idx = np.random.choice(sample_idx, size = batch_size, replace = True)

        rna_raw = self.rna[sample_idx,:].X.copy()
        rna_raw = self._sparse_array(rna_raw)

        '''
        ori data inference
        '''
        
        self.rna_model.eval()
        self.atac_model.eval()

        X = torch.from_numpy(rna_raw).float().to(self.device)
        infer_dataset = infer_dataset(X)
        infer_loader = DataLoader(infer_dataset, batch_size = 128, shuffle = False)

        embed_list = []
        recon_list = []
        with torch.no_grad():
            for batch in infer_loader:
                batch = batch.to(self.device)
                z, z_mu, z_logvar, z_embed = self.rna_model.get_embedding(batch)
                atac_recon = self.atac_model.decoder(z_embed)

                embed_list.append(z_mu.detach().cpu().numpy())
                recon_list.append(atac_recon.detach().cpu().numpy())
        raw_embed = np.concatenate(embed_list)
        raw_recon = np.concatenate(recon_list)

        '''
        up perturb data inference
        '''
        mean_tf = X[:,tf_id].mean().item()

        X[:,tf_id] = 

        infer_dataset = infer_dataset(X)
        infer_loader = DataLoader(infer_dataset, batch_size = 128, shuffle = False)

        embed_list = []
        recon_list = []
        with torch.no_grad():
            for batch in infer_loader:
                z, z_mu, z_logvar, z_embed = self.rna_model.get_embedding(batch)
                atac_recon = self.atac_model.decoder(z_embed)

                embed_list.append(z_mu.detach().cpu().numpy())
                recon_list.append(atac_recon.detach().cpu().numpy())
        down_embed = np.concatenate(embed_list)
        down_recon = np.concatenate(recon_list)

        '''
        down perturb data inference
        '''
        X[:,tf_id] = 0

        infer_dataset = infer_dataset(X)
        infer_loader = DataLoader(infer_dataset, batch_size = 128, shuffle = False)

        embed_list = []
        recon_list = []
        with torch.no_grad():
            for batch in infer_loader:
                z, z_mu, z_logvar, z_embed = self.rna_model.get_embedding(batch)
                atac_recon = self.atac_model.decoder(z_embed)

                embed_list.append(z_mu.detach().cpu().numpy())
                recon_list.append(atac_recon.detach().cpu().numpy())
        down_embed = np.concatenate(embed_list)
        down_recon = np.concatenate(recon_list)


        # rna_raw = self._sparse_array(rna_raw)
        # atac_raw = self.atac[sample_idx,:].X
        # rna_raw = self._sparse_array(rna_raw)
        # atac_raw = self._sparse_array(atac_raw)


        
    
        