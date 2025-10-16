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
    drop_id = rna.var.interval == 'NA'
    rna = rna[:, ~drop_id]
    atac = mdata['peak_count']

    '''
    generate the train id and test id using int index
    '''
    IDX = np.arange(rna.shape[0])
    train_id = IDX[rna.obs.split == 'train']
    test_id = IDX[rna.obs.split == 'test'] 

    '''
    create the scpair data
    '''
    adata_paired = merge_paired_data([rna, atac])

    train_id, val_id = train_test_split(train_id, test_size = 0.1, random_state=42)
    print(len(train_id), len(val_id), len(test_id))

    train_id = rna.obs.index[train_id]
    train_id = train_id.tolist()

    val_id = rna.obs.index[val_id]
    val_id = val_id.tolist()

    test_id = rna.obs.index[test_id]
    test_id = test_id.tolist()
    pre_split = [train_id, val_id, test_id]

    """s
    perform the split operation for the multi-modal object
    """
    adata_paired = training_split(adata_paired, pre_split=[train_id, val_id, test_id])
    print(adata_paired.obs.scPair_split.value_counts()) 
    return adata_paired