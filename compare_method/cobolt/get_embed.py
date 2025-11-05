import torch 
import numpy as np
from torch.utils.data import DataLoader, Dataset 
import scanpy as sc
import pandas as pd
import anndata as ad
import muon as mu

from cobolt.utils import SingleData, MultiomicDataset
from cobolt.model import Cobolt
import sys 
# sys.path.append('compare_method/cobolt')
from load_data import *
import torch 
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from scipy import sparse
from typing import List
from xgboost import XGBRegressor
from tqdm import tqdm
import anndata as ad 
import os

def collate_wrapper(batch, omic_combn):
    dataset = [x[1] for x in batch]
    batch = [x[0] for x in batch]
    dataset = [torch.tensor(list(x)) if include else None
               for x, include in zip(zip(*dataset), omic_combn)]
    batch = [torch.from_numpy(sparse.vstack(x).toarray()).float() if include else None
             for x, include in zip(zip(*batch), omic_combn)]
    return batch, dataset
def modified_get_latent(model,
                        omic_combn,
                        data="train",
                        what="latent",
                        batch_size = 128,
                        return_barcode=False):
    if data == "train":
        sample_idx = model.train_idx
    elif data == "test":
        sample_idx = model.test_idx
    else:
        raise ValueError

    if model.epoch == 0:
        raise Exception("Model haven't been trained yet.")

    sample_idx = np.intersect1d(model.dataset.get_comb_idx(omic_combn), sample_idx)
    dl = DataLoader(
        dataset=Subset(model.dataset, sample_idx),
        batch_size=batch_size,
        collate_fn=lambda x: collate_wrapper(x, omic_combn),
        shuffle=False
    )
    latent = []
    for i, x in tqdm(enumerate(dl)):
        x = [[x_i.to(model.device) if x_i is not None else None for x_i in y] for y in x]
        if what == "latent":
            latent += [model.model.get_latent(x, elbo_bool=omic_combn)]
        elif what == "topic_prop":
            latent += [model.model.get_topic_prop(x, elbo_bool=omic_combn)]
    res = np.concatenate(latent)
    if return_barcode:
        return res, model.dataset.get_barcode()[sample_idx]
    return res

def modified_calc_all_latent(model, batch_size = 128,
                    target: List[bool] = None):
    """
    Calculate the latent variable estimation.

    Parameters
    ----------
    target
        A list of boolean indicating which posterior distribution is used
        as benchmark for correction.
    """
    n_modality = len(model.dataset.omic)
    if target is None:
        target = [True] * n_modality
    target_dt, target_barcode = modified_get_latent(model, target, batch_size = batch_size, return_barcode=True)
    dt_corrected = [target_dt]
    barcode_corrected = target_barcode
    print('============================')
    for i, x in enumerate(model.dataset.omic):
        om_combn = [False] * n_modality
        om_combn[i] = True
        raw_dt, raw_barcode = modified_get_latent(model, om_combn, batch_size = batch_size, return_barcode=True)
        bool_train = np.isin(raw_barcode, target_barcode)
        bool_test = ~np.isin(raw_barcode, barcode_corrected)
        if sum(bool_test) != 0:
            raw_dt_train = raw_dt[bool_train, ]
            raw_dt_test = raw_dt[bool_test]
            raw_bc_train = raw_barcode[bool_train]
            raw_bc_test = raw_barcode[bool_test]
            barcode_dict = {x: i for i, x in enumerate(raw_bc_train)}
            reorder = [barcode_dict[i] for i in target_barcode]
            raw_dt_train = raw_dt_train[reorder, ]
            this_predicted = []
            for i in range(model.n_latent):
                xgb_model = XGBRegressor()
                xgb_model.fit(X=raw_dt_train, y=target_dt[:, i].copy())
                this_predicted.append(xgb_model.predict(raw_dt_test))
            dt_corrected.append(np.asarray(this_predicted).T)
            barcode_corrected = np.concatenate((barcode_corrected, raw_bc_test))
    dt_corrected = np.vstack(dt_corrected)
    dt_corrected = (dt_corrected.T - np.mean(dt_corrected, axis=1)).T
    model.latent = {
        "latent": dt_corrected,
        "barcode": barcode_corrected,
        "epoch": model.epoch
    }
    return model 


case_id = 'case_12'
train_data_path = f'/home/rsun@ZHANGroup.local/solid-recover/data/{case_id}/train_count.h5mu'
test_data_path = f'/home/rsun@ZHANGroup.local/solid-recover/data/{case_id}/test_count.h5mu'
model_ckpt = f'/home/rsun@ZHANGroup.local/solid-recover/compare_method/cobolt/{case_id}/model.pth'
save_dir = f'/home/rsun@ZHANGroup.local/solid-recover/compare_method/cobolt/{case_id}'
if __name__ == '__main__':

    'set cobolt data'
    multi_dt = load_multi_dt(train_data_path, test_data_path)   

    ## set model and train

    model = Cobolt(dataset = multi_dt, lr = 1e-4, n_latent = 16)
    model.model = torch.load(model_ckpt, weights_only=False)
    model.epoch = 200
    model = modified_calc_all_latent(model, 100)

    # get rna embed and atac embed in test data
    test_atac_id = []
    test_rna_id = []

    for ele in model.latent['barcode']:
        if 'test_atac' in ele:
            test_atac_id.append(True)
        else:
            test_atac_id.append(False)
        if 'test_rna' in ele:
            test_rna_id.append(True) 
        else:
            test_rna_id.append(False)

    atac_embed = model.latent['latent'][test_atac_id]
    rna_embed = model.latent['latent'][test_rna_id] 

    N = rna_embed.shape[0]
    sc_test = ad.AnnData(X = np.random.randn(N*2, 10),)

    sc_test.obs.loc[:,'batch'] = ['rna']*N + ['atac']*N
    sc_test.obs.loc[:,'idx'] = np.concatenate([np.arange(N), np.arange(N)])
    sc_test.obsm['X_embed'] = np.concatenate([rna_embed, atac_embed])
    
    sc_test.write_h5ad(os.path.join(save_dir, f'model_embed.h5ad'))
    print('Program Over')

    adata = ad.AnnData(X = model.latent['latent'])
    adata.obs.index = model.latent['barcode']
    adata.write(os.path.join(save_dir, 'cobolt_latent.h5ad'))
