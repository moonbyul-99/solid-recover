import scanpy as sc 
import numpy as np
import pandas as pd
import anndata as ad
import os 
import matplotlib.pyplot as plt
import json 
import sys 
sys.path.append('../../src')
from metrics import * 
import torch


# import necessary packages for single cell analysis
import os
import copy
import scipy
import random
import numpy as np
import pandas as pd
from scipy import sparse
import muon as mu

import anndata
import scanpy as sc
import scvi
import sys
sys.path.append('/home/rsun@ZHANGroup.local/projects_list/scPair')
from scpair import *

from sklearn.model_selection import train_test_split 
from datetime import datetime
import os 
import anndata as ad
from load_scpair_data import load_data  

from torch.utils.data import Dataset, DataLoader 
class infer_data(Dataset):
    def __init__(self, rna, atac):
        self.rna = rna
        self.atac = atac
    
    def __len__(self):
        return self.rna.shape[0]
    
    def __getitem__(self, idx):
        return self.rna[idx,], self.atac[idx,]

def pipe(train_data_path, test_data_path, save_dir,  batch_size = 128, device = 'cuda', save = True):
    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_1/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_1/test_count.h5mu'

    # save_dir = 'pred_eval'
    # os.makedirs(save_dir, exist_ok=True)

    adata_paired = load_data(train_data_path, test_data_path)
    """
    set up scPair object
    """
    scpair_setup = scPair_object(scobj = adata_paired, cov=None, modalities = {'Gene Expression': 'zinb', 'Peaks': 'ber'},
                            sample_factor_rna=True, sample_factor_atac=False, infer_library_size_rna=False, infer_library_size_atac=True,
                            batchnorm=True, layernorm=True, SEED=0, hidden_layer=[800, 30], dropout_rate=0.1, learning_rate_prediction=1e-3, 
                            max_epochs=1000,save_path = save_dir,)

    pseudo_save_path = 'pseudo_save'
    os.makedirs(pseudo_save_path, exist_ok=True)
    scpair_setup = scPair_object(scobj = adata_paired, cov=None, modalities = {'Gene Expression': 'zinb', 'Peaks': 'ber'},
                            sample_factor_rna=True, sample_factor_atac=False, infer_library_size_rna=False, infer_library_size_atac=True,
                            batchnorm=True, layernorm=True, SEED=0, hidden_layer=[800, 30], dropout_rate=0.1, learning_rate_prediction=1e-3, 
                            max_epochs=1,save_path = pseudo_save_path,)
    scpair_setup.data_loader_builder()
    
    '''
    Load checkpoint file
    '''


    ckpt_dir = f'/home/rsun@ZHANGroup.local/solid-recover/compare_method/scpair/{save_dir}'

    decoder_g2p_path = os.path.join(ckpt_dir, 'decoder_Gene Expression_to_Peaks.pt')
    decoder_p2g_path = os.path.join(ckpt_dir, 'decoder_Peaks_to_Gene Expression.pt')
    encoder_g2p_path = os.path.join(ckpt_dir, 'encoder_Gene Expression_to_Peaks.pt')
    encoder_p2g_path = os.path.join(ckpt_dir, 'encoder_Peaks_to_Gene Expression.pt')
    mapping_g2p_path = os.path.join(ckpt_dir, 'mapping_Gene Expression_to_Peaks.pt')
    mapping_p2g_path = os.path.join(ckpt_dir, 'mapping_Peaks_to_Gene Expression.pt')

    decoder_g2p = torch.load(decoder_g2p_path)
    decoder_p2g = torch.load(decoder_p2g_path)
    encoder_g2p = torch.load(encoder_g2p_path)
    encoder_p2g = torch.load(encoder_p2g_path)
    mapping_g2p = torch.load(mapping_g2p_path)
    mapping_p2g = torch.load(mapping_p2g_path)



    scpair_setup.encoder_dict['Gene Expression to Peaks'] = encoder_g2p
    scpair_setup.encoder_dict['Peaks to Gene Expression'] = encoder_p2g
    scpair_setup.decoder_dict['Gene Expression to Peaks'] = decoder_g2p
    scpair_setup.decoder_dict['Peaks to Gene Expression'] = decoder_p2g

    scpair_setup.mapping_dict['Gene Expression_to_Peaks'] = mapping_g2p
    scpair_setup.mapping_dict['Peaks_to_Gene Expression'] = mapping_p2g

    '''
    perform prediction
    '''
    low_dim_embeddings, _ = scpair_setup.reference_embeddings()
    low_dim_embeddings_mapped, _ = scpair_setup.mapped_embeddings()
    predictions = scpair_setup.predict_test()
    rna_pred = predictions['Gene Expression_test']
    atac_pred = predictions['Peaks_test']
    # predictions.keys()
    test_id = adata_paired.obs.scPair_split  == 'test'
    adata_test = adata_paired[test_id]

    atac_id = adata_test.var.modality == 'Peaks'
    rna_id = adata_test.var.modality == 'Gene Expression'

    '''
    save prediction
    '''
    eval_dir = os.path.join(save_dir, 'pred_result')
    os.makedirs(eval_dir, exist_ok=True)
    rna_test = adata_test[:, rna_id]
    rna_test.write(os.path.join(eval_dir, 'rna_raw.h5ad'))
    atac_test = adata_test[:, atac_id]
    atac_test.write(os.path.join(eval_dir, 'atac_raw.h5ad'))

    rna_pred = ad.AnnData(rna_pred, obs = rna_test.obs, var = rna_test.var)
    rna_pred.write(os.path.join(eval_dir, 'rna_pred.h5ad'))
    atac_pred = ad.AnnData(atac_pred, obs = atac_test.obs, var = atac_test.var)
    atac_pred.write(os.path.join(eval_dir, 'atac_pred.h5ad'))

    print('OVER')

if __name__ == '__main__':

    for i in [8,9,11,12]:
        train_data_path = f'/home/rsun@ZHANGroup.local/solid-recover/data/case_{i}/train_count.h5mu'
        test_data_path = f'/home/rsun@ZHANGroup.local/solid-recover/data/case_{i}/test_count.h5mu'
        save_dir = f'case_{i}'

        pipe(train_data_path, test_data_path, save_dir, batch_size = 128, device = 'cuda', save = True)
        # pipe(train_data_path, test_data_path, save_dir)