# import sys 
# sys.path.append('../src')
from sr_model import *
from sr_dataset import *
import os
import numpy as np 
import pandas as pd 
from typing import List, Dict, Union, Any 
import muon as mu 
import scanpy as sc

class eval_data:
    def __init__(self, data_path: str, train_split_path: str, test_split_path: str):
        '''
        Args: 
        data_path: str, path to data, the mdata is required to be count matrix
        split_path: str, path to split .npy file
        '''

        self.data_path = data_path

        self.train_idx = np.load(train_split_path)
        self.test_idx = np.load(test_split_path)
        print(f'train size: {len(self.train_idx)}, test size: {len(self.test_idx)}')
    
    def load_pair_data(self, key_1: str, key_2: str):
        mdata = mu.read_h5mu(self.data_path)

        data_1 = mdata[key_1]
        data_2 = mdata[key_2]
        self.data_1 = data_1
        self.data_2 = data_2 

    def data_qc(self,):

        print(f'data 1 shape {self.data_1.shape}')
        sc.pp.filter_genes(self.data_1, min_cells=int(self.data_1.shape[0] * 0.01))
        print(f'data 1 shape after qc {self.data_1.shape}')

        print(f'data 1 shape {self.data_2.shape}')
        sc.pp.filter_genes(self.data_2, min_cells=int(self.data_2.shape[0] * 0.01))
        print(f'data 1 shape after qc {self.data_2.shape}')

    def data_normalize(self):
        sc.pp.normalize_total(self.data_1, target_sum=1e4)
        sc.pp.log1p(self.data_1) 

        sc.pp.normalize_total(self.data_2, target_sum=1e4)
        sc.pp.log1p(self.data_2) 

    def get_train_test_dataset(self):

        train_data_1, train_data_2 = self.data_1[self.train_idx,:], self.data_2[self.train_idx,:]
        test_data_1, test_data_2 = self.data_1[self.test_idx,:], self.data_2[self.test_idx,:]

        train_dataset = pair_data(Base_sr._adata_format(train_data_1), Base_sr._adata_format(train_data_2))
        test_dataset = pair_data(Base_sr._adata_format(test_data_1), Base_sr._adata_format(test_data_2))
        return train_dataset, test_dataset
    

def data_prepare( data_path: str, train_split_path: str, test_split_path: str, key_1: str, key_2: str):

    eval_data_obj = eval_data(data_path, train_split_path, test_split_path)
    eval_data_obj.load_pair_data(key_1, key_2)
    eval_data_obj.data_qc()
    eval_data_obj.data_normalize()
    train_dataset, test_dataset = eval_data_obj.get_train_test_dataset()
    return train_dataset, test_dataset 


# if __name__ == '__main__':
    # print('test load eval data script')

    # print('case 1====================================================================================')

    # data_path = '../../sr_project/eval_data/merge_dataset/mdata.h5mu'
    # train_split_path = '../../sr_project/eval_data/data_split/train_id_4.npy'
    # test_split_path = '../../sr_project/eval_data/data_split/test_id_4.npy'

    # key_1 = 'rna_count'
    # key_2 = 'peak_count'
    # train_dataset, test_dataset = data_prepare(data_path, train_split_path, test_split_path, key_1, key_2)

    # print( 'case 2==================================================================================')
    # data_path = '../../sr_project/eval_data/merge_dataset/mdata.h5mu'
    # train_split_path = '../../sr_project/eval_data/data_split/train_id_1.npy'
    # test_split_path = '../../sr_project/eval_data/data_split/test_id_1.npy'

    # key_1 = 'rna_count'
    # key_2 = 'ga_count'
    # train_dataset, test_dataset = data_prepare(data_path, train_split_path, test_split_path, key_1, key_2)


    # print( 'case 3==================================================================================')
    # data_path = '../../sr_project/eval_data/merge_dataset/kidney.h5mu'
    # train_split_path = '../../sr_project/eval_data/data_split/train_id_kidney.npy'
    # test_split_path = '../../sr_project/eval_data/data_split/test_id_kidney.npy'

    # key_1 = 'rna_count'
    # key_2 = 'ga_count'
    # train_dataset, test_dataset = data_prepare(data_path, train_split_path, test_split_path, key_1, key_2)

    # print('TEST OVER')

    


