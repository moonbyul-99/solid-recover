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
from typing import List

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
            X = X.toarray().astype(np.float32)
            return X
        
    def _rna2atac(self,X):
        '''给定rna数据，生成atac数据'''
        self.rna_model.eval()
        self.atac_model.eval()

        dataset = infer_dataset(X)
        infer_loader = DataLoader(dataset, batch_size = 128, shuffle = False)

        embed_list = []
        recon_list = []
        with torch.no_grad():
            for batch in infer_loader:
                batch = batch.to(self.device)
                z, z_mu, z_logvar, z_embed = self.rna_model.get_embedding(batch)
                atac_recon = self.atac_model.decoder(z_embed)

                embed_list.append(z_mu.detach().cpu().numpy())
                recon_list.append(atac_recon.detach().cpu().numpy())
        embed = np.concatenate(embed_list)
        recon = np.concatenate(recon_list)
        return {'embed': embed, 'recon': recon}
    
    def _atac2rna(self,X):
        '''给定atac数据，生成rna数据'''
        self.rna_model.eval()
        self.atac_model.eval()

        dataset = infer_dataset(X)
        infer_loader = DataLoader(dataset, batch_size = 128, shuffle = False)

        embed_list = []
        recon_list = []
        with torch.no_grad():
            for batch in infer_loader:
                batch = batch.to(self.device)
                z, z_mu, z_logvar, z_embed = self.atac_model.get_embedding(batch)
                atac_recon = self.rna_model.decoder(z_embed)

                embed_list.append(z_mu.detach().cpu().numpy())
                recon_list.append(atac_recon.detach().cpu().numpy())
        embed = np.concatenate(embed_list)
        recon = np.concatenate(recon_list)
        return {'embed': embed, 'recon': recon}

    def tf_re_perturb(self, tf_names:List[str], cluster_id: str,  rate: float = 0.0):
        '''
        获取对给定的TFs, 对某个cluster中的细胞，通过调低或者调高后对下游的REs 的影响

        tf_name: str, name of the transcription factor
        rate: float, 调整给定的TF表达为 rate* TF_expression 
        '''
        for tf in tf_names:
            if tf not in self.rna.var.index:
                print(f'{tf} not in rna data, break')
                return None
        if cluster_id not in self.cluster_info:
            print(f'{cluster_id} not in cluster info, break')
            return None 


        '''获取每个tf对应的输入维度'''
        tf_ids = []
        for tf in tf_names:
            tf_id = self.rna.var.index.get_loc(tf)
            tf_ids.append(tf_id)
        
        '''获取这个cluster 中的细胞'''
        sample_idx = self.cluster_info == cluster_id
        rna_raw = self.rna[sample_idx,:].X.copy()
        rna_raw = self._sparse_array(rna_raw)  #np.array
        
        '''获取原始数据下对应的RE 开放程度'''
        raw_res = self._rna2atac(rna_raw)

        '''获取扰动数据下对应的RE 开放程度'''
        rna_perturb = self.rna[sample_idx,:].X.copy()
        rna_perturb = self._sparse_array(rna_perturb)
        rna_perturb[:, tf_ids] = (rna_perturb[:, tf_ids] + 1) * rate
        pert_res = self._rna2atac(rna_perturb)
        return {'raw_embed': raw_res['embed'],
                'raw_pred': raw_res['recon'],
                'pert_embed': pert_res['embed'],
                'pert_pred': pert_res['recon']}
    
    def re_tg_perturb(self, re_names:List[str], cluster_id: str,  rate: float = 0.0):
        '''
        获取对给定的REs, 对某个cluster中的细胞，通过调低或者调高后对下游的REs 的影响

        tf_name: str, name of the transcription factor
        rate: float, 调整给定的TF表达为 rate* TF_expression 
        '''
        for re in re_names:
            if re not in self.atac.var.index:
                print(f'{re} not in atac data, break')
                return None
        if cluster_id not in self.cluster_info:
            print(f'{cluster_id} not in cluster info, break')
            return None 
        
        '''获取每个re对应的输入维度'''
        re_ids = []
        for re in re_names:
            re_id = self.atac.var.index.get_loc(re)
            re_ids.append(re_id)
        
        '''获取这个cluster 中的细胞'''
        sample_idx = self.cluster_info == cluster_id
        atac_raw = self.atac[sample_idx,:].X.copy()
        atac_raw = self._sparse_array(atac_raw)  #np.array
        
        '''获取原始数据下对应的RE 开放程度'''
        raw_res = self._atac2rna(atac_raw)

        '''获取扰动数据下对应的RE 开放程度'''
        atac_perturb = self.atac[sample_idx,:].X.copy()
        atac_perturb = self._sparse_array(atac_perturb)
        atac_perturb[:, re_ids] = (atac_perturb[:, re_ids] + 1) * rate
        pert_res = self._atac2rna(atac_perturb)
        return {'raw_embed': raw_res['embed'],
                'raw_pred': raw_res['recon'],
                'pert_embed': pert_res['embed'],
                'pert_pred': pert_res['recon']}
        


        
    
        