import numpy as np 
import pandas as pd 
import snapatac2 as snap
import os 
import scanpy as sc
import anndata as ad 
import json
import torch
from datasets import load_from_disk, concatenate_datasets
from torch.utils.data import Dataset, DataLoader 


class single_data(Dataset):
    '''
    Define the dataset for single cell omic data
    '''

    def __init__(self,
                 data):
        self.data = data 
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return {'feature': self.data[index, :]}

class omic_data(Dataset):
    """
    Define the dataset for single cell paired omics data,
    currently only support RNA-seq and gene activity data.
    """
    def __init__(self, rna, ga):
        self.rna = rna 
        self.ga = ga 

    def __len__(self):
        return self.rna.shape[0]
    
    def __getitem__(self,idx):
        return {'rna': self.rna[idx,:],
                'ga': self.ga[idx,:]}
