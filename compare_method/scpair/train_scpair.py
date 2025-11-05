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

def train_scpair(train_data_path, test_data_path, save_dir):
    os.makedirs(save_dir, exist_ok = True)

    adata_paired = load_data(train_data_path, test_data_path)
    """
    set up scPair object
    """
    scpair_setup = scPair_object(scobj = adata_paired, cov=None, modalities = {'Gene Expression': 'zinb', 'Peaks': 'ber'},
                            sample_factor_rna=True, sample_factor_atac=False, infer_library_size_rna=False, infer_library_size_atac=True,
                            batchnorm=True, layernorm=True, SEED=0, hidden_layer=[800, 30], dropout_rate=0.1, learning_rate_prediction=1e-3, 
                            max_epochs=1000,save_path = save_dir,)

    """
    start running optimization for scPair framework
    """
    res = scpair_setup.run()

    """
    extrct the learned embeddings
    """
    e, e_df = scpair_setup.reference_embeddings()
    me, me_df = scpair_setup.mapped_embeddings()
    e_df.keys() # dict_keys(['Gene Expression_train', 'Gene Expression_val', 'Gene Expression_test', 'Peaks_train', 'Peaks_val', 'Peaks_test'])
    me_df.keys() # dict_keys(['Gene Expression to Peaks_train', 'Gene Expression to Peaks_val', 'Gene Expression to Peaks_test', 'Peaks to Gene Expression_train', 'Peaks to Gene Expression_val', 'Peaks to Gene Expression_test'])


    # save result 
    gene_test = e['Gene Expression_test']
    peak_test = e['Peaks_test'] 

    p2g_test = me['Peaks to Gene Expression_test']
    g2p_test = me['Gene Expression to Peaks_test']

    N = gene_test.shape[0]

    scdata = ad.AnnData(X = np.random.rand(N,10))
    scdata.obsm['gene'] = gene_test.numpy()
    scdata.obsm['peak'] = peak_test.numpy()
    scdata.obsm['p2g'] = p2g_test.numpy()
    scdata.obsm['g2p'] = g2p_test.numpy()

    scdata.write_h5ad(os.path.join(save_dir, 'scdata.h5ad'))
    print(f'program over')
    return None

if __name__ == '__main__':
    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_1/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_1/test_count.h5mu'
    # save_dir = 'case_1'

    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_2/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_2/test_count.h5mu'
    # save_dir = 'case_2'
    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_3/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_3/test_count.h5mu'
    # save_dir = 'case_3'

    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_4/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_4/test_count.h5mu'
    # save_dir = 'case_4'

    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_5/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_5/test_count.h5mu'
    # save_dir = 'case_5'

    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_6/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_6/test_count.h5mu'
    # save_dir = 'case_6'


    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_8/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_8/test_count.h5mu'
    # save_dir = 'case_8'

    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_9/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_9/test_count.h5mu'
    # save_dir = 'case_9'
    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_10/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_10/test_count.h5mu'
    # save_dir = 'case_10'

    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_11/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_11/test_count.h5mu'
    # save_dir = 'case_11'

    train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_12/train_count.h5mu'
    test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_12/test_count.h5mu'
    save_dir = 'case_12'
    train_scpair(train_data_path , test_data_path, save_dir)