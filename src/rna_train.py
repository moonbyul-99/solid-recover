import sys
import os
from datasets import load_from_disk, concatenate_datasets
from datasets import Dataset
import numpy as np 
import pandas as pd 
from torch.utils.data import DataLoader  
import yaml
from sr_model import single_sr 

def main(config_path):
    """
    prepare dataset
    """

    dataset = load_from_disk('/home/rsun@ZHANGroup.local/rna_pretrain/hf_data/allen_23_training_new')
    dataset_dic = dataset.train_test_split(test_size = 0.025, seed = 42)
    train_dataset, test_dataset = dataset_dic['train'], dataset_dic['test']

    B = 2048 # batch size
    train_loader = DataLoader(train_dataset,
                            batch_size = B,
                            num_workers= 4,
                            shuffle = True)

    test_loader = DataLoader(test_dataset,
                            batch_size = B,
                            #num_workers = 4
                            shuffle = False) 
    
    """
    prepare config
    """

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(config)
    
    """
    define sr_model and train
    """ 

    rna_model = single_sr(config)
    rna_model.train_model(train_loader,
                            test_loader)

if __name__ == '__main__': 
    if len(sys.argv) != 2:
        print("Usage: python rna_train.py <config_path>")
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)