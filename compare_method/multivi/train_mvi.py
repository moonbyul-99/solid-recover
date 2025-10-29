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

from load_data import *
import warnings 
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
from pytorch_lightning.loggers import TensorBoardLogger


def train_mvi(train_data_path, test_data_path, save_dir):

    '''
    step 1: load data
    '''
    adata_mvi = prepare_mvi_data(train_data_path, test_data_path)

    '''
    step 2: train model
    ''' 

    scvi.model.MULTIVI.setup_anndata(adata_mvi, batch_key="modality")

    model = scvi.model.MULTIVI(
        adata_mvi,
        n_genes=(adata_mvi.var["modality"] == "Gene Expression").sum(),
        n_regions=(adata_mvi.var["modality"] == "Peaks").sum(),
    )

    model.view_anndata_setup()

    # === 设置日志目录 ===
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # === 创建 logger ===
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name='',        # 避免再创建子目录
        version='',     # 避免 version_0 子目录
        default_hp_metric=False  # 不记录默认的 hp_metric
    )

    model.train( logger = logger)  ## Use default parameters 

    '''
    step 3: save model and data
    '''

    # get all_latent embedding 

    total_rna = model.get_latent_representation(modality = 'expression')
    total_atac =  model.get_latent_representation(modality = 'accessibility')
    total_joint =  model.get_latent_representation(modality = 'joint')

    adata_mvi.obsm['X_total_rna'] = total_rna
    adata_mvi.obsm['X_total_atac'] = total_atac
    adata_mvi.obsm['X_total_joint'] = total_joint  

    # save model and save embedding 

    model.save(save_dir,overwrite = True) 

    adata_mvi.write(f'{save_dir}/adata_mvi.h5ad')
    print('Program Over')

if __name__ == '__main__':

    '''
    case 1 training 
    '''
    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_1/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_1/test_count.h5mu'
    # save_dir = 'case_1'
    # train_mvi(train_data_path , test_data_path, save_dir) 

    '''
    case 2 training
    '''
    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_2/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_2/test_count.h5mu'
    # save_dir = 'case_2'
    # train_mvi(train_data_path , test_data_path, save_dir) 

    '''
    case 3 training
    '''
    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_3/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_3/test_count.h5mu'
    # save_dir = 'case_3'
    # train_mvi(train_data_path , test_data_path, save_dir) 


    # '''
    # case 4 training 
    # '''
    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_4/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_4/test_count.h5mu'
    # save_dir = 'case_4'
    # '''
    # case 5 training 
    # '''
    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_5/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_5/test_count.h5mu'
    # save_dir = 'case_5_count'

    # '''
    # case 6 training 
    # '''
    # train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_6/train_count.h5mu'
    # test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_6/test_count.h5mu'
    # save_dir = 'case_6'

    '''
    case 8 training 
    '''
    train_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_8/train_count.h5mu'
    test_data_path = '/home/rsun@ZHANGroup.local/solid-recover/data/case_8/test_count.h5mu'
    save_dir = 'case_8'
    train_mvi(train_data_path , test_data_path, save_dir)