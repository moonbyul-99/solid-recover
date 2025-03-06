import sys
import os
import numpy as np 
import pandas as pd 
from torch.utils.data import DataLoader  
import yaml
from sr_model import paired_sr, single_sr
from dataset import omic_data,single_data
import torch
import muon as mu 
import json 
import scanpy as sc 

'''
train a paired model without pretraining weights
'''
def load_data():
    mdata = mu.read_h5mu('/home/rsun@ZHANGroup.local/multi_pretrain/evaluation/notebook/eval_data/M_rna_1/mdata.h5mu')
    
    rna = mdata['rna_count']
    gadata = mdata['ga_count'] 

    mutual_gene = np.load('/home/rsun@ZHANGroup.local/atac_pretrain/src/mutual_gene.npy', allow_pickle=True)
    gadata = gadata[:,mutual_gene]
    
    # lo1p transform 

    sc.pp.normalize_total(gadata, target_sum= 1e4)
    sc.pp.log1p(gadata)

    sc.pp.normalize_total(rna, target_sum= 1e4)
    sc.pp.log1p(rna)

    # feature selection

    sc.pp.highly_variable_genes(rna, n_top_genes= 5000)
    rna = rna[:,rna.var['highly_variable']]

    sc.pp.highly_variable_genes(gadata, n_top_genes= 10000)
    gadata = gadata[:,gadata.var['highly_variable']] 

    print(rna.shape, gadata.shape)

    #train_idx, test_idx = train_test_split(np.arange(N), test_size = 0.1, random_state = 42)
    train_idx = np.load('/home/rsun@ZHANGroup.local/multi_pretrain/evaluation/sr_result/train_test_split/train_id_1.npy', allow_pickle = True)
    test_idx = np.load('/home/rsun@ZHANGroup.local/multi_pretrain/evaluation/sr_result/train_test_split/test_id_1.npy', allow_pickle = True)

    rna_train, rna_test = rna[train_idx,:], rna[test_idx,:]
    gadata_train, gadata_test = gadata[train_idx,:], gadata[test_idx,:]
    return rna_train, rna_test, gadata_train, gadata_test

def process_data(rna_train, rna_test, gadata_train, gadata_test):

    train_rna = rna_train.X.toarray().astype(np.float32)
    train_ga = gadata_train.X.toarray().astype(np.float32)
    train_rna = torch.from_numpy(train_rna)
    train_ga = torch.from_numpy(train_ga)

    test_rna = rna_test.X.toarray().astype(np.float32)
    test_ga = gadata_test.X.toarray().astype(np.float32)
    test_rna = torch.from_numpy(test_rna)
    test_ga = torch.from_numpy(test_ga)
    return train_rna, train_ga, test_rna, test_ga

def set_rna_config(N, B=1024):
    # N is the dataset size , B is the batch size 

    if N >= 100000:
        print('Large dataset, please use pair_train.py')
        return None 
    steps = int(40*N/B) # at least 1000 steps 


    config = {
        'omic': {'model_type': 'rna'},

        'network': {
            'feature_num': 5000,
            'hidden_dims': [512, 256, 128],
            'dropout': 0.1,
            'layernorm_eps': 1e-8,
            'activation': 'leaky_relu',
            'input_dropout': 0.2,
            'vae_weight': 0,
            'ce_weights': None,
            'class_dict': None
        },
        'optimizer': {
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'warmup_steps': 100,
            'anneal_steps': steps,
            'min_lr': 1e-6
        },
        'training': {
            'device': 'cuda',
            'training_steps': steps,
            'eval_steps': 100000,
            'save_steps': 100000, # set large, do not save checkpoint during training
            'log_dir': 'logs',
            'save_dir': 'saved_models',
            'run_name': 'tiny_rna'
        }
    }
    return config

def set_ga_config(N, B=1024):
    # N is the dataset size , B is the batch size 

    if N >= 100000:
        print('Large dataset, please use pair_train.py')
        return None 
    steps = int(60*N/B) # at least 1000 steps 


    config = {
        'omic': {'model_type': 'ga'},

        'network': {
            'feature_num': 10000,
            'hidden_dims': [512, 256, 128],
            'dropout': 0.1,
            'layernorm_eps': 1e-8,
            'activation': 'leaky_relu',
            'input_dropout': 0.2,
            'vae_weight': 0,
            'ce_weights': None,
            'class_dict': None
        },
        'optimizer': {
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'warmup_steps': 100,
            'anneal_steps': steps,
            'min_lr': 1e-6
        },
        'training': {
            'device': 'cuda',
            'training_steps': steps,
            'eval_steps': 100000,
            'save_steps': 100000, # set large, do not save checkpoint during training
            'log_dir': 'logs',
            'save_dir': 'saved_models',
            'run_name': 'tiny_ga'
        }
    }
    return config
def sr_rna_train(rna_data):
    rna_dataset = single_data(rna_data)

    rna_loader = DataLoader(rna_dataset, batch_size = 1024, shuffle= True)

    '''
    ini config
    '''
    config = set_rna_config(N = rna_data.shape[0], B = 1024)

    '''
    set rna model 
    '''
    rna_model = single_sr(config)
    rna_model.train_model(rna_loader, save_config= False)
    return rna_model, config

def sr_ga_train(ga_data):
    ga_dataset = single_data(ga_data)
    ga_loader = DataLoader(ga_dataset, batch_size = 1024, shuffle= True)

    '''
    ini config
    '''
    config = set_ga_config(N = ga_data.shape[0], B = 1024)

    '''
    set rna model 
    '''
    ga_model = single_sr(config)
    ga_model.set_optimizer()
    ga_model.train_model(ga_loader, save_config=False)
    return ga_model, config 

def paired_train(paired_config,
                 rna_config,
                 ga_config,
                 sr_rna, 
                 sr_ga,
                 train_loader, 
                 val_loader):

    paired_model = paired_sr(paired_config, 
                             rna_config, 
                             ga_config,
                             sr_rna,
                             sr_ga)
    paired_model.train_model(train_loader,
                            val_loader, save_config=False)
    return paired_model


def main(config_path):
    """
    prepare paired config
    """

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(config)
    print('config over')

    """
    prepare dataset
    """

    rna_train, rna_test, gadata_train, gadata_test = load_data()
    train_rna, train_ga, test_rna, test_ga = process_data(rna_train, rna_test, gadata_train, gadata_test)
    print(train_rna.shape, train_ga.shape)
    print(test_rna.shape, test_ga.shape)
    """
    prepare single omic dataset, for single omic data, we use all data to train the single omic model
    """

    combined_rna = torch.cat((train_rna, test_rna), dim=0)
    combined_ga = torch.cat((train_ga, test_ga), dim=0)
    print(combined_rna.shape, combined_ga.shape)

    #rna_dataset = single_data(combined_rna)
    #ga_dataset = single_data(combined_ga)

    """
    prepare  multiomic data
    """
    traindata = omic_data(train_rna, train_ga)
    testdata = omic_data(test_rna, test_ga) 
    print('dataset over')

    '''
    prepare dataloader 
    '''
    if 'batchsize' not in config: 
        batchsize = 2048
    else:
        batchsize = config['batchsize']

    train_loader = DataLoader(traindata, batch_size= batchsize, shuffle=True)
    test_loader = DataLoader(testdata, batch_size= batchsize, shuffle=True)
    print('dataloader over')
    
    '''
    define single_sr model and train 
    '''
    # Run rna_train and ga_train in parallel
    #from concurrent.futures import ProcessPoolExecutor

    #with ProcessPoolExecutor(max_workers=2) as executor:
    #    future_rna = executor.submit(rna_train, combined_rna)
    #    future_ga = executor.submit(ga_train, combined_ga)

    #    sr_rna, rna_config = future_rna.result()
    #    sr_ga, ga_config = future_ga.result()
    sr_rna, rna_config = sr_rna_train(combined_rna)
    print('rna over')
    sr_ga, ga_config = sr_ga_train(combined_ga)
    print('ga over')

    """
    define sr_model and train
    """ 

    #paired_model = paired_sr(config)
    #paired_model.train_model(train_loader,
                            #test_loader)
    paired_train(config,
                 rna_config,
                 ga_config,
                 sr_rna, 
                 sr_ga,
                 train_loader, 
                 val_loader = test_loader)
    print('OVER')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python rna_train.py <config_path>")
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)