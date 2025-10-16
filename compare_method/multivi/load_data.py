import pandas as pd 
import torch.nn.functional as F
import numpy as np
import muon as mu 
import scanpy as sc
import yaml 
import torch 
from torch.utils.data import Dataset, DataLoader 
import torch.nn as nn 
import os 
import argparse
import matplotlib.pyplot as plt
import scvi
import anndata as ad
from mudata import MuData
from scipy.sparse import hstack 

def load_mdata(train_data_path, test_data_path):
    '''
    Load the train and test data, concat to mdata
    '''

    train_mdata = mu.read_h5mu(train_data_path)
    test_mdata = mu.read_h5mu(test_data_path)


    '''
    concat to mdata
    '''
    res = {}
    for key in train_mdata.mod_names:
        adata = ad.concat([train_mdata.mod[key], test_mdata.mod[key]])
        adata.obs.loc[:,'split'] = ['train']*train_mdata.mod[key].shape[0] + ['test']*test_mdata.mod[key].shape[0]
        adata.var = train_mdata.mod[key].var.copy()
        res[key] = adata 
    mdata = MuData(res)

    '''
    save only rna and atac data, filter rna var with NA interval
    '''
    rna = mdata['rna_count']
    atac = mdata['peak_count']

    '''filter rna var with NA interval'''
    drop_id = rna.var.interval == 'NA'
    rna = rna[:, ~drop_id]

    mdata = MuData({'rna_count': rna, 'peak_count': atac})
    return mdata 

def prepare_var(rna, atac):

    '''
    modify the var info to satisfy mvi requirements
    '''
    rna_var = rna.var.copy()
    atac_var = atac.var.copy()

    '''
    generate new atac var
    '''
    chr_list = []
    start_list = []
    end_list = []
    for ele in atac.var.index:
        chr,_ = ele.split(':')[0], ele.split(':')[1]
        start, end = int(_.split('-')[0]), int(_.split('-')[1])
        chr_list.append(chr)
        start_list.append(start)
        end_list.append(end)

    new_atac_var = atac_var.copy()
    new_atac_var.loc[:,'ID'] = new_atac_var.index.values
    new_atac_var.loc[:,'modality'] = 'Peaks'
    new_atac_var.loc[:,'chr'] = chr_list
    new_atac_var.loc[:,'start'] = start_list 
    new_atac_var.loc[:,'end'] = end_list


    '''
    generate new rna var
    '''
    chr_list = []
    start_list = []
    end_list = []

    for ele in rna.var.interval:
        chr,_ = ele.split(':')[0], ele.split(':')[1]
        start, end = int(_.split('-')[0]), int(_.split('-')[1])
        chr_list.append(chr)
        start_list.append(start)
        end_list.append(end)

    new_rna_var = rna_var.copy()
    new_rna_var.loc[:,'ID'] = rna_var.loc[:,'gene_ids']
    new_rna_var.loc[:,'modality'] = 'Gene Expression'
    new_rna_var.loc[:,'chr'] = chr_list
    new_rna_var.loc[:,'start'] = start_list
    new_rna_var.loc[:,'end'] = end_list

    new_rna_var = new_rna_var.drop(columns = ['gene_ids', 'feature_types','genome','interval'])
    mvi_var = pd.concat([new_rna_var, new_atac_var], axis = 0)
    return mvi_var, new_rna_var, new_atac_var

def prepare_mvi_data(train_data_path, test_data_path):

    '''
    return the mvi data for multivi training
    '''

    mdata = load_mdata(train_data_path, test_data_path)
    rna = mdata['rna_count']
    atac = mdata['peak_count']

    mvi_var, new_rna_var, new_atac_var = prepare_var(rna, atac)

    '''
    multivi data prepare, add test atac and test rna to mvidata
    '''


    rna_X = mdata['rna_count'].X 
    atac_X = mdata['peak_count'].X

    X = hstack([rna_X, atac_X]) 

    train_idx = mdata['rna_count'].obs.split == 'train'
    test_idx = mdata['rna_count'].obs.split == 'test'
    print(train_idx.sum(), test_idx.sum())

    paired_X = X[train_idx]
    test_atac_X = X[test_idx]
    test_rna_X = X[test_idx]

    rna_feature_idx = np.arange(mvi_var.shape[0])[mvi_var.loc[:,'modality'] == 'Gene Expression']
    test_atac_X[:, rna_feature_idx] = 0

    atac_feature_idx =np.arange(mvi_var.shape[0])[mvi_var.loc[:,'modality'] == 'Peaks']
    test_rna_X[:, atac_feature_idx] = 0

    paired_data = ad.AnnData(paired_X, obs = mdata['rna_count'].obs.loc[train_idx,:], var = mvi_var)

    test_atac = ad.AnnData(test_atac_X, mdata['rna_count'].obs.loc[test_idx,:], var = mvi_var)
    test_atac.obs.index = 'atac_' + test_atac.obs.index.astype(str) 

    test_rna = ad.AnnData(test_rna_X, obs = mdata['rna_count'].obs.loc[test_idx,:], var = mvi_var)
    test_rna.obs.index = 'rna_' + test_rna.obs.index.astype(str) 

    # We can now use the organizing method from scvi to concatenate these anndata
    adata_mvi = scvi.data.organize_multiome_anndatas(rna_anndata = test_rna, 
                                                    multi_anndata = paired_data,         
                                                    atac_anndata = test_atac)
    adata_mvi = adata_mvi[:, adata_mvi.var["modality"].argsort()].copy()
    return adata_mvi

