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
from multiprocessing import Process 
from load_scb_data import *
from torch.utils.data import Dataset, DataLoader 
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ad.ImplicitModificationWarning)

class infer_data(Dataset):
    def __init__(self, rna, atac):
        self.rna = rna 
        self.atac = atac 
    
    def __len__(self):
        return self.rna.shape[0]
    
    def __getitem__(self, idx):
        return self.rna[idx,:], self.atac[idx,:]

def pipe(train_data_path, test_data_path, save_dir, batch_size = 128, device = 'cuda', save = True):
    '''
    set model 
    '''

    print('Step 1: Set model initialization parameters')

    RNA_data, ATAC_data, test_id, train_id, chrom_list = load_data(train_data_path, test_data_path)
    # os.makedirs(save_dir, exist_ok=True)

    '''
    set model
    '''
    RNA_input_dim = RNA_data.X.shape[1]#len([i for i in RNA_data.var['highly_variable'] if i])
    ATAC_input_dim = ATAC_data.X.shape[1]

    R_kl_div = 1 / RNA_input_dim * 20
    A_kl_div = 1 / ATAC_input_dim * 20
    kl_div = R_kl_div + A_kl_div

    model = Model(
        R_encoder_nlayer = 2,
        A_encoder_nlayer = 2,
        R_decoder_nlayer = 2,
        A_decoder_nlayer = 2,
        R_encoder_dim_list = [RNA_input_dim, 256, 128],
        A_encoder_dim_list = [ATAC_input_dim, 32 * len(chrom_list), 128],
        R_decoder_dim_list = [128, 256, RNA_input_dim],
        A_decoder_dim_list = [128, 32 * len(chrom_list), ATAC_input_dim],
        R_encoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
        A_encoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
        R_decoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
        A_decoder_act_list = [nn.LeakyReLU(), nn.Sigmoid()],
        translator_embed_dim = 128,
        translator_input_dim_r = 128,
        translator_input_dim_a = 128,
        translator_embed_act_list = [nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()],
        discriminator_nlayer = 1,
        discriminator_dim_list_R = [128],
        discriminator_dim_list_A = [128],
        discriminator_act_list = [nn.Sigmoid()],
        dropout_rate = 0.1,
        R_noise_rate = 0.5,
        A_noise_rate = 0.3,
        chrom_list = chrom_list,
        logging_path = None,
        RNA_data = RNA_data,
        ATAC_data = ATAC_data
    )


    '''load model ckpt'''
    print('Step 2: Load model checkpoint')

    model.RNA_encoder.load_state_dict(torch.load(os.path.join(save_dir,'model','RNA_encoder.pt')))
    model.ATAC_encoder.load_state_dict(torch.load(os.path.join(save_dir,'model','ATAC_encoder.pt')))
    model.RNA_decoder.load_state_dict(torch.load(os.path.join(save_dir,'model','RNA_decoder.pt')))
    model.ATAC_decoder.load_state_dict(torch.load(os.path.join(save_dir,'model','ATAC_decoder.pt')))
    model.R_translator.load_state_dict(torch.load(os.path.join(save_dir,'model','R_translator.pt')))
    model.A_translator.load_state_dict(torch.load(os.path.join(save_dir,'model','A_translator.pt'))) 
    model.translator.load_state_dict(torch.load(os.path.join(save_dir,'model','translator.pt'))) 
    model.discriminator_A.load_state_dict(torch.load(os.path.join(save_dir,'model','discriminator_A.pt')))
    model.discriminator_R.load_state_dict(torch.load(os.path.join(save_dir,'model','discriminator_R.pt'))) 


    '''
    generate data
    '''
    atac_split = model.ATAC_data_obs.split == 'test'
    rna_split = model.RNA_data_obs.split == 'test'
    assert (model.ATAC_data_obs.index[atac_split] == model.RNA_data_obs.index[rna_split]).all() 

    rna_data = model.RNA_data[rna_split,:].astype(np.float32)
    atac_data = model.ATAC_data[atac_split, :].astype(np.float32)
 
    print('Step 3: print create inference dataset')



        
    infer_dataset= infer_data(rna_data, atac_data)
    infer_loader = DataLoader(infer_dataset, batch_size = batch_size, shuffle = False)
    rna_predict = []
    atac_predict = []

    print('Step 4: generate inferene data')
    for batch in infer_loader:
        RNA_input = batch[0].to(device)
        ATAC_input = batch[1].to(device)

        R2 = model.RNA_encoder(RNA_input)
        A2 = model.ATAC_encoder(ATAC_input)

        R2R, R2A, mu_r, sigma_r = model.translator.test_model(R2, 'RNA')
        A2R, A2A, mu_a, sigma_a = model.translator.test_model(A2, 'ATAC')        

        # R2R = model.RNA_decoder(R2R)
        R2A = model.ATAC_decoder(R2A)
        A2R = model.RNA_decoder(A2R)
        # A2A = model.ATAC_decoder(A2A) 

        rna_predict.append(A2R.detach().to('cpu').numpy())
        atac_predict.append(R2A.detach().to('cpu').numpy())

    rna_predict = np.concatenate(rna_predict)
    atac_predict = np.concatenate(atac_predict)

    rna_pred = ad.AnnData(rna_predict, obs = model.RNA_data_obs.loc[rna_split, :], var = model.RNA_data_var)
    atac_pred = ad.AnnData(atac_predict, obs = model.ATAC_data_obs.loc[atac_split,:], var = model.ATAC_data_var)
    
    rna_raw = ad.AnnData(rna_data, obs = model.RNA_data_obs.loc[rna_split, :], var = model.RNA_data_var )
    atac_raw = ad.AnnData(atac_data, obs = model.ATAC_data_obs.loc[atac_split,:], var = model.ATAC_data_var)
    if save:
        pred_dir = os.path.join(save_dir,'pred_result')
        os.makedirs(pred_dir, exist_ok=True)
        rna_pred.write(os.path.join(pred_dir,'rna_pred.h5ad'))
        atac_pred.write(os.path.join(pred_dir,'atac_pred.h5ad'))
        rna_raw.write(os.path.join(pred_dir,'rna_raw.h5ad'))
        atac_raw.write(os.path.join(pred_dir,'atac_raw.h5ad'))

    return rna_pred, atac_pred, rna_raw, atac_raw 


# from scipy.stats import pearsonr 
# def pearson_corr_columns_scipy(A, B):
#     p = A.shape[1]
#     corrs = np.empty(p)
#     for i in range(p):
#         corrs[i], _ = pearsonr(A[:, i], B[:, i])
#     return corrs
# def simple_eval(rna_pred, atac_pred, rna_raw, atac_raw):
#     corrs = pearson_corr_columns_scipy(rna_pred.X, rna_raw.X)

if __name__ == '__main__':
    for i in [8,12]:#[8,9,11,12]:
        train_data_path = f'/home/rsun@ZHANGroup.local/solid-recover/data/case_{i}/train_count.h5mu'
        test_data_path = f'/home/rsun@ZHANGroup.local/solid-recover/data/case_{i}/test_count.h5mu'
        #save_dir = f'/home/rsun@ZHANGroup.local/solid-recover/compare_method/scb/case_{i}'
        save_dir = f'/home/rsun@ZHANGroup.local/solid-recover/compare_method/scb/case_{i}_new'
        pipe(train_data_path, test_data_path, save_dir, batch_size = 128, device = 'cuda', save = True)
        print(f'CASE{i} DONE')