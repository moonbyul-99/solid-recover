import numpy as np 
import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader 


class single_data(Dataset):
    '''
    Define the dataset for single cell omic data
    '''
    def __init__(self, data):
        '''
        Args:
            data: input single cell omic data, require to be torch.tensor with dtype float32
        '''

        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Input data must be a torch.Tensor, but got {type(data)}")
        
        if data.dtype != torch.float32:
            raise TypeError(f"Input data dtype must be torch.float32, but got {data.dtype}")
        
        if torch.isnan(data).any():
            raise ValueError("Input data contains NaN values.")
        
        self.data = data
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return {'feature': self.data[index, :]}

class pair_data(Dataset):
    """
    Define the dataset for single cell paired omics data, currently only support two omics data
    """
    def __init__(self, omic_1, omic_2):

        '''
        Args: 
            omic_1: the first omic data, require to be torch.tensor with dtype float32 
            omic_2: the second omic data,require to be torch.tensor with dtype float32
        '''
        self._format_check(omic_1, 'omic_1')
        self._format_check(omic_2, 'omic_2')

        if omic_1.shape[0] != omic_2.shape[0]:
            raise ValueError('omic_1 and omic_2 must have the same number of samples')

        self.omic_1 = omic_1
        self.omic_2 = omic_2
    @staticmethod
    def _format_check(data, omic_key):
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"{omic_key} input data must be a torch.Tensor, but got {type(data)}")
        
        if data.dtype != torch.float32:
            raise TypeError(f"{omic_key} input data dtype must be torch.float32, but got {data.dtype}")
        
        if torch.isnan(data).any():
            raise ValueError(f"{omic_key} input data contains NaN values.")

    def __len__(self):
        return self.omic_1.shape[0]
    
    def __getitem__(self,idx):
        return {'omic_1': self.omic_1[idx,:],
                'omic_2': self.omic_2[idx,:]}
    
    def to_gpu(self):
        self.omic_1 = self.omic_1.to('cuda')
        self.omic_2 = self.omic_2.to('cuda')
    

