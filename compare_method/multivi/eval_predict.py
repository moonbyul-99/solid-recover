import numpy as  np 
import pandas as pd
import scanpy as sc 
import muon as mu 
import pandas as pd 
import torch.nn.functional as F
import numpy as np
import muon as mu 
import anndata as ad
import yaml 
import torch 
from torch.utils.data import Dataset, DataLoader 
import torch.nn as nn 
import os 
import argparse
import matplotlib.pyplot as plt

import scvi

from load_data import *
import warnings 
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
from pytorch_lightning.loggers import TensorBoardLogger 


def pipe(train_data_path, test_data_path, save_dir):

    '''
    step 1: load data
    '''
    adata_mvi = prepare_mvi_data(train_data_path, test_data_path)

    '''
    step 2: load model weights
    '''
    model = scvi.model.MULTIVI.load(save_dir, adata = adata_mvi)

    '''
    step 3: generate predictions
    '''
    test_idx = adata_mvi.obs.split.values == 'test'
    atac_idx = adata_mvi.obs.modality.values == 'accessibility'
    rna_idx = adata_mvi.obs.modality.values == 'expression' 
    all_indices = np.arange(adata_mvi.shape[0])

    test_rna = all_indices[rna_idx & test_idx]
    test_atac = all_indices[atac_idx & test_idx]

    '''generate rna predictions'''
    res = model.get_normalized_expression(indices = test_rna)
    pred_rna = res.values
    rna_var =  adata_mvi.var_names[:model.n_genes]
    rna_index = res.index.values 
    for i,ele in enumerate(rna_index):
        ele = ele[4:]
        # ele.replace('_expression','')
        rna_index[i] = ele.replace('_expression','')
    rna_pred = ad.AnnData(pred_rna,)
    rna_pred.var.index = rna_var 
    rna_pred.obs.index = rna_index

    '''generate atac predictions'''
    res = model.get_accessibility_estimates(adata_mvi, indices = test_atac, normalize_cells = True)
    pred_atac = res.values
    atac_var = adata_mvi.var_names[model.n_genes:]
    atac_index = res.index.values 
    for i,ele in enumerate(atac_index):
        ele = ele[5:]  ## remove the first atac_
        atac_index[i] = ele.replace('_accessibility','')
    atac_pred = ad.AnnData(pred_atac,)
    atac_pred.var.index = atac_var 
    atac_pred.obs.index = atac_index

    '''step 4: save predictions'''
    pred_dir = os.path.join(save_dir,'pred_result')
    os.makedirs(pred_dir, exist_ok=True)

    rna_pred.write(os.path.join(pred_dir,'rna_pred.h5ad'))
    atac_pred.write(os.path.join(pred_dir,'atac_pred.h5ad'))
    print('Program Over')


if __name__ == '__main__':
    for i in [8,9,11]:#,12]:
        train_data_path = f'/home/rsun@ZHANGroup.local/solid-recover/data/case_{i}/train_count.h5mu'
        test_data_path = f'/home/rsun@ZHANGroup.local/solid-recover/data/case_{i}/test_count.h5mu'
        save_dir = f'/home/rsun@ZHANGroup.local/solid-recover/compare_method/multivi/case_{i}'
        pipe(train_data_path, test_data_path, save_dir)