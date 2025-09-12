import sys 
sys.path.append('src')
from sr_model import *

import os
from datasets import load_from_disk, concatenate_datasets
from datasets import Dataset
import numpy as np 
import pandas as pd 
from torch.utils.data import DataLoader  


if __name__ == '__main__':
    dataset = load_from_disk('/home/rsun@ZHANGroup.local/rna_pretrain/hf_data/allen_23_training_new')
    dataset_dic = dataset.train_test_split(test_size = 0.025, seed = 42)

    train_dataset, test_dataset = dataset_dic['train'], dataset_dic['test']

    feature_num = dataset[0]['feature'].shape[0]
    hidden_params= {'hidden_dim' : 1024,
                    'block_num' : 8} 
    embed_dim = 128
    use_rmsnorm = True
    use_residual = True
    dropout_p = 0
    vae_model = True

    project_dir = 'runs/rna_demo'
    train_steps = 500
    eval_points = 50
    save_points = 300

    rna_model = single_sr(feature_num, hidden_params, embed_dim, use_rmsnorm, use_residual, dropout_p, vae_model)
    rna_model.set_dataset(train_dataset, test_dataset)
    rna_model.set_dataloader(batch_size = 1024)
    rna_model.set_loss(beta = 1)
    rna_model.set_optimizer(lr = 1e-3, warmup_steps=100, steady_1_steps = 100, cosine_anneal_steps=100, min_lr = 1e-6)
    rna_model.set_project(project_dir)
    rna_model.train_model( train_steps,eval_points,save_points,device = 'cuda')

    print('OVER')