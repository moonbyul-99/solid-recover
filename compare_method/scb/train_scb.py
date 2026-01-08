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


def train_scb(train_data_path, test_data_path, save_dir):
    
    '''
    load data
    '''
    RNA_data, ATAC_data, test_id, train_id, chrom_list = load_data(train_data_path, test_data_path)
    os.makedirs(save_dir, exist_ok=True)

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

    # set train_id, val_id, test_id

    train_id, valid_id = train_test_split(train_id, test_size=0.1, random_state=42)
    train_id_r = train_id
    train_id_a = train_id
    validation_id_r = valid_id
    validation_id_a = valid_id
    test_id_r = test_id
    test_id_a = test_id

    os.makedirs(os.path.join(save_dir,'log'), exist_ok=True)
    model.train(
        R_encoder_lr = 0.001,
        A_encoder_lr = 0.001,
        R_decoder_lr = 0.001,
        A_decoder_lr = 0.001,
        R_translator_lr = 0.001,
        A_translator_lr = 0.001,
        translator_lr = 0.001,
        discriminator_lr = 0.005,
        R2R_pretrain_epoch = 100,  #
        A2A_pretrain_epoch = 100,  #
        lock_encoder_and_decoder = False,
        translator_epoch = 200,    #
        patience = 50,
        batch_size = 128,
        r_loss = nn.MSELoss(size_average=True),
        a_loss = nn.BCELoss(size_average=True),
        d_loss = nn.BCELoss(size_average=True),
        loss_weight = [1, 2, 1, R_kl_div, A_kl_div, kl_div],
        train_id_r = train_id_r,
        train_id_a = train_id_a,
        validation_id_r = validation_id_r,
        validation_id_a = validation_id_a,
        output_path = save_dir,
        seed = 19193,
        kl_mean = True,
        R_pretrain_kl_warmup = 50,
        A_pretrain_kl_warmup = 50,
        translation_kl_warmup = 50,
        load_model = None,
        logging_path = os.path.join(save_dir,'log')
    )

    ## generate test result
    test_rna = RNA_data[test_id_r]
    test_atac = ATAC_data[test_id_a]

    test_rna = test_rna.X.toarray().astype(np.float32)
    test_atac = test_atac.X.toarray().astype(np.float32)

    test_rna = torch.from_numpy(test_rna)
    test_atac = torch.from_numpy(test_atac) 

    A2R_list = []
    A2A_list = []

    R2R_list = []
    R2A_list = []

    N = test_rna.shape[0]
    for i in range(0, N, 128):
        X = test_rna[i : min(i + 128,N),:].to('cuda')
        Y = test_atac[i : min(i + 128,N),:].to('cuda')

        A2 = model.ATAC_encoder(Y)
        A2R, A2A, mu_a, sigma_a = model.translator.test_model(A2, 'ATAC')

        R2 = model.RNA_encoder(X)
        R2R, R2A, mu_r, sigma_r = model.translator.test_model(R2, 'RNA')

        A2R_list.append(A2R.detach().cpu().numpy())
        A2A_list.append(A2A.detach().cpu().numpy())
        R2R_list.append(R2R.detach().cpu().numpy())
        R2A_list.append(R2A.detach().cpu().numpy())
    
    A2R = np.concatenate(A2R_list)
    A2A = np.concatenate(A2A_list)
    R2R = np.concatenate(R2R_list)
    R2A = np.concatenate(R2A_list)

    sc_test = ad.AnnData(X = np.random.randn(N,10))
    sc_test.obsm['A2R'] = A2R
    sc_test.obsm['R2R'] = R2R
    sc_test.obsm['R2A'] = R2A
    sc_test.obsm['A2A'] = A2A

    sc_test.write_h5ad(os.path.join(save_dir, 'sc_test.h5ad'))
    print('scb Train OVER')
    return None 

if __name__ == '__main__':
    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_1/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_1/test_count.h5mu'
    # save_dir = 'case_1'

    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_4/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_4/test_count.h5mu'
    # save_dir = 'case_4'
    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_2/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_2/test_count.h5mu'
    # save_dir = 'case_2'
    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_3/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_3/test_count.h5mu'
    # save_dir = 'case_3'

    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_5/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_5/test_count.h5mu'
    # save_dir = 'case_5'


    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_6/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_6/test_count.h5mu'
    # save_dir = 'case_6'

    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_8/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_8/test_count.h5mu'
    # save_dir = 'case_8_new'

    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_9/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_9/test_count.h5mu'
    # save_dir = 'case_9'


    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_10/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_10/test_count.h5mu'
    # save_dir = 'case_10'

    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_11/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_11/test_count.h5mu'
    # save_dir = 'case_11_new'

    train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_12/train_count.h5mu'
    test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_12/test_count.h5mu'
    save_dir = 'case_12_new'
    train_scb(train_data_path , test_data_path, save_dir)
