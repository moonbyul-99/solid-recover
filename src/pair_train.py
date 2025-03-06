import sys
import os
import numpy as np 
import pandas as pd 
from torch.utils.data import DataLoader  
import yaml
from sr_project.src.sr_model_new import single_sr, paired_sr
from dataset import omic_data 
import torch
import muon as mu 
import json 
import scanpy as sc 
from sr_net import multi_model
def load_data():
    mdata = mu.read_h5mu('/home/rsun@ZHANGroup.local/multi_pretrain/evaluation/notebook/eval_data/M_rna_1/mdata.h5mu')
    
    rna = mdata['rna_count']
    gadata = mdata['ga_count'] 

    # feature selection of gadata 
    #feature_path = '/home/rsun@ZHANGroup.local/atac_pretrain/hf_data/ga_feature.json' 

    #with open(feature_path, 'r') as f:
    #    res = json.load(f)
    #f.close()
    #gadata = gadata[:,res['feature_names']]
    mutual_gene = np.load('/home/rsun@ZHANGroup.local/atac_pretrain/src/mutual_gene.npy', allow_pickle=True)
    gadata = gadata[:,mutual_gene]
    
    # lo1p transform 

    sc.pp.normalize_total(gadata, target_sum= 1e4)
    sc.pp.log1p(gadata)

    sc.pp.normalize_total(rna, target_sum= 1e4)
    sc.pp.log1p(rna)

    # train test split
    #N = mdata.shape[0]

    #train_idx, test_idx = train_test_split(np.arange(N), test_size = 0.1, random_state = 42)
    train_idx = np.load('/home/rsun@ZHANGroup.local/multi_pretrain/evaluation/sr_result/train_test_split/train_id_1.npy', allow_pickle = True)
    test_idx = np.load('/home/rsun@ZHANGroup.local/multi_pretrain/evaluation/sr_result/train_test_split/test_id_1.npy', allow_pickle = True)

    rna_train, rna_test = rna[train_idx,:], rna[test_idx,:]
    gadata_train, gadata_test = gadata[train_idx,:], gadata[test_idx,:]
    return rna_train, rna_test, gadata_train, gadata_test

def main(config_path):
    """
    prepare config
    """

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(config)
    print('config over')

    """
    prepare dataset
    """

    rna_train, rna_test, gadata_train, gadata_test = load_data()

    train_rna = rna_train.X.toarray().astype(np.float32)
    train_ga = gadata_train.X.toarray().astype(np.float32)
    train_rna = torch.from_numpy(train_rna)
    train_ga = torch.from_numpy(train_ga)

    test_rna = rna_test.X.toarray().astype(np.float32)
    test_ga = gadata_test.X.toarray().astype(np.float32)
    test_rna = torch.from_numpy(test_rna)
    test_ga = torch.from_numpy(test_ga)


    print(train_rna.shape, train_ga.shape)
    print(test_rna.shape, test_ga.shape)

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
    
    """
    define sr_model and train
    """ 

    paired_model = paired_sr(config)
    paired_model.train_model(train_loader,
                            test_loader)
    print('OVER')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python rna_train.py <config_path>")
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)