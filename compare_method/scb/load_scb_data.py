import os
from scipy import io
import pandas as pd
import muon as mu 
import scanpy as sc 
import numpy as np
import torch
import anndata as ad
import sys 
sys.path.append('/home/rsun@ZHANGroup.local/projects_list/scButterfly/scButterfly')
from data_processing import RNA_data_preprocessing, ATAC_data_preprocessing
from split_datasets import *
from train_model import Model
import torch.nn as nn
from sklearn.model_selection import train_test_split
from datetime import datetime
import muon as mu 
from mudata import MuData


def load_data(train_data_path, test_data_path):
    '''
    Load the train and test data, concat to mdata
    '''

    train_mdata = mu.read_h5mu(train_data_path)
    test_mdata = mu.read_h5mu(test_data_path)


    '''
    concat to mdata
    '''
    res = {}
    for key in ['rna_count', 'peak_count']:
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

    #mdata = MuData({'rna_count': rna, 'peak_count': atac})
    
    '''
    generate the train id and test id using int index
    '''
    IDX = np.arange(rna.shape[0])
    train_id = IDX[rna.obs.split == 'train']
    test_id = IDX[rna.obs.split == 'test']

    # drop peak not in chr 
    peak_sel = []
    for ele in atac.var.index.values:
        if 'chr' in ele:
            peak_sel.append(True)
        else:
            peak_sel.append(False)
    atac = atac[:,peak_sel]
   
    ## perform scb preprocessing 

    print('perform scb RNA preprocessing')
    RNA_data = RNA_data_preprocessing(
        rna,
        normalize_total=True,
        log1p=True,
        use_hvg=True,
        n_top_genes=3000,
        save_data=False,
        file_path=None,
        logging_path=None
        )
    
    print('perform scb ATAC preprocessing')

    ATAC_data = ATAC_data_preprocessing(
        atac,
        binary_data=True,
        filter_features=True,
        fpeaks=0.005,
        tfidf=False,  #TIME CONSUMING, VERY LOW EFFICENCY IMPLEMENTATION
        normalize=True,
        save_data=False,
        file_path=None,
        logging_path=None
    )[0]

    ## scb chrom preprocess 
    print('Additional SCB atac preprocessing')

    chrom = []
    for ele in ATAC_data.var.index:
        a = ele.split(':')[0]
        if 'chr' in a:
            chrom.append(a)
        else:
            chrom.append('NA')
    ATAC_data.var['chrom'] = chrom

    chrom_list = []
    last_one = ''
    for i in range(len(ATAC_data.var.chrom)):
        temp = ATAC_data.var.chrom[i]
        if temp[0 : 3] == 'chr':
            if not temp == last_one:
                chrom_list.append(1)
                last_one = temp
            else:
                chrom_list[-1] += 1
        else:
            chrom_list[-1] += 1

    print(chrom_list, end="")

    return RNA_data, ATAC_data, test_id, train_id, chrom_list