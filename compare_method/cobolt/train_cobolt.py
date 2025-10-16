from cobolt.utils import SingleData, MultiomicDataset
from cobolt.model import Cobolt
import os
from scipy import io
import pandas as pd
import muon as mu 
import scanpy as sc 
import numpy as np
import torch
from datetime import datetime
import json
import anndata as ad 
from load_data import * 

def train_cobolt(train_data_path, test_data_path, save_dir):
    os.makedirs(save_dir, exist_ok = True)

    multi_dt = load_multi_dt(train_data_path, test_data_path)   

    ## set model and train

    model = Cobolt(dataset = multi_dt, lr = 1e-3, n_latent = 16)
    model.train(num_epochs = 200)

    ## get model training loss 
    loss = model.history
    with open(os.path.join(save_dir, 'loss.json'), 'w') as f:
        json.dump(loss, f)

    ## save model
    model_path = os.path.join(save_dir, 'model.pth')
    torch.save(model.model, model_path)    

    # get latent embed
    model.calc_all_latent()
    latent = model.get_all_latent()

    # get rna embed and atac embed in test data
    test_atac_id = []
    test_rna_id = []

    for ele in latent[-1]:
        if 'test_atac' in ele:
            test_atac_id.append(True)
        else:
            test_atac_id.append(False)
        if 'test_rna' in ele:
            test_rna_id.append(True) 
        else:
            test_rna_id.append(False)

    atac_embed = latent[0][test_atac_id]
    rna_embed = latent[0][test_rna_id] 

    N = rna_embed.shape[0]
    sc_test = ad.AnnData(X = np.random.randn(N*2, 10),)

    sc_test.obs.loc[:,'batch'] = ['rna']*N + ['atac']*N
    sc_test.obs.loc[:,'idx'] = np.concatenate([np.arange(N), np.arange(N)])
    sc_test.obsm['X_embed'] = np.concatenate([rna_embed, atac_embed])
    sc_test.write_h5ad(os.path.join(save_dir, f'model_embed.h5ad'))
    print('Program Over')

if __name__ == '__main__':

    train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_1/train_count.h5mu'
    test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_1/test_count.h5mu'
    save_dir = 'case_1'

    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_4/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_4/test_count.h5mu'
    # save_dir = 'case_4'
    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_2/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_2/test_count.h5mu'
    # save_dir = 'case_2'

    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_3/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_3/test_count.h5mu'
    # save_dir = 'case_3'
    train_cobolt(train_data_path ,test_data_path ,save_dir)