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
from anndata import AnnData



def _qc(adata: AnnData):
    print(f'data before qc {adata.shape}')
    sc.pp.filter_genes(adata, min_cells = int(0.01*adata.shape[0]))
    print(f'data qfter qc {adata.shape}')

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata) 
    return adata

def _filter_data(train_adata: AnnData, test_adata: AnnData):
    print('train data qc')
    train_adata = _qc(train_adata)
    print('test data filter')
    test_adata = test_adata[:, train_adata.var.index]
    sc.pp.normalize_total(test_adata, target_sum=1e4)
    sc.pp.log1p(test_adata) 

    print(f'OVER, train data size: {train_adata.shape}, test data size: {test_adata.shape}')
    return train_adata, test_adata

def split_data(data_path: str, train_split_path: str, test_split_path: str, save_dir: str):
    #mdata = mu.read_h5mu('/home/rsun@ZHANGroup.local/sr_project/eval_data/merge_dataset/mdata.h5mu')
    os.makedirs(save_dir, exist_ok=True)

    mdata = mu.read_h5mu(data_path)
    train_idx = np.load(train_split_path)
    test_idx = np.load(test_split_path)

    mdata_train = mdata[train_idx]
    mdata_test = mdata[test_idx]

    train_dic = {}
    test_dic = {}

    for key in mdata.mod_names:
        train_adata = mdata_train[key]
        test_adata = mdata_test[key]

        train_adata, test_adata = _filter_data(train_adata, test_adata)

        train_dic[key] = train_adata
        test_dic[key] = test_adata
    
    mdata_train = mu.MuData(train_dic)
    mdata_test = mu.MuData(test_dic)

    mu.write_h5mu(os.path.join(save_dir, 'train.h5mu'),mdata_train)
    mu.write_h5mu(os.path.join(save_dir, 'test.h5mu'),mdata_test)
    print('OVER')
    
def data_prepare(train_data_path: str, test_data_path: str, key_1: str, key_2: str, to_gpu: bool = False):

    '''
    prepare the paired data for sr training
    load train and test multi omic data
    construct the train and test pair dataset
    '''

    train_data = mu.read_h5mu(train_data_path)
    test_data = mu.read_h5mu(test_data_path)

    train_data_1 = train_data[key_1]
    train_data_2 = train_data[key_2]

    test_data_1 = test_data[key_1]
    test_data_2 = test_data[key_2]

    train_dataset = pair_data(Base_sr._adata_format(train_data_1), Base_sr._adata_format(train_data_2))
    test_dataset = pair_data(Base_sr._adata_format(test_data_1), Base_sr._adata_format(test_data_2))
    
    if to_gpu:
        train_dataset.to_gpu()
        test_dataset.to_gpu()
    return train_dataset, test_dataset





import multiprocessing as mp
from functools import partial


if __name__ == '__main__':
    data_path = '/home/rsun@ZHANGroup.local/sr_project/eval_data/merge_dataset/mdata.h5mu'
    train_split_path_list = [f'/home/rsun@ZHANGroup.local/sr_project/eval_data/data_split/train_id_{i}.npy' for i in range(1, 5)]
    test_split_path_list = [f'/home/rsun@ZHANGroup.local/sr_project/eval_data/data_split/test_id_{i}.npy' for i in range(1, 5)]
    save_dir_list = [f'/home/rsun@ZHANGroup.local/solid-recover/data/case_{i}' for i in range(1, 5)]

    args_list = list(zip(train_split_path_list, test_split_path_list, save_dir_list))
    worker_func = partial(split_data, data_path)

    with mp.Pool(processes=4) as pool:
        pool.starmap(worker_func, args_list)







#*************************************************************************************************************************************************************
#*************************************************************************************************************************************************************
#*************************************************************************************************************************************************************
#*************************************************************************************************************************************************************


# 这部分代码在初始阶段使用，后续评测中弃用，主要存在如下问题：
# 1. 效率：每次评测截断都要重走一遍QC
# 2. 潜在的data leakage: 现在全部数据集上过滤特征，之后再分割数据集

# 2025.10.12修改目标：
# 1. 弃用当前流程，将不同的测试数据集单独保存在data目录下，做好训练集测试集划分后直接读取
# 2. 先划分数据集，再过滤特征





# class eval_data:
#     def __init__(self, data_path: str, train_split_path: str, test_split_path: str):
#         '''
#         Args: 
#         data_path: str, path to data, the mdata is required to be count matrix
#         split_path: str, path to split .npy file
#         '''

#         self.data_path = data_path

#         self.train_idx = np.load(train_split_path)
#         self.test_idx = np.load(test_split_path)
#         print(f'train size: {len(self.train_idx)}, test size: {len(self.test_idx)}')
    
#     def load_pair_data(self, key_1: str, key_2: str):
#         mdata = mu.read_h5mu(self.data_path)

#         data_1 = mdata[key_1]
#         data_2 = mdata[key_2]
#         self.data_1 = data_1
#         self.data_2 = data_2 
    

#     def data_qc(self,):

#         print(f'data 1 shape {self.data_1.shape}')
#         sc.pp.filter_genes(self.data_1, min_cells=int(self.data_1.shape[0] * 0.01))
#         print(f'data 1 shape after qc {self.data_1.shape}')

#         print(f'data 1 shape {self.data_2.shape}')
#         sc.pp.filter_genes(self.data_2, min_cells=int(self.data_2.shape[0] * 0.01))
#         print(f'data 1 shape after qc {self.data_2.shape}')

#     def data_normalize(self):
#         sc.pp.normalize_total(self.data_1, target_sum=1e4)
#         sc.pp.log1p(self.data_1) 

#         sc.pp.normalize_total(self.data_2, target_sum=1e4)
#         sc.pp.log1p(self.data_2) 

#     def get_train_test_dataset(self):

#         train_data_1, train_data_2 = self.data_1[self.train_idx,:], self.data_2[self.train_idx,:]
#         test_data_1, test_data_2 = self.data_1[self.test_idx,:], self.data_2[self.test_idx,:]

#         train_dataset = pair_data(Base_sr._adata_format(train_data_1), Base_sr._adata_format(train_data_2))
#         test_dataset = pair_data(Base_sr._adata_format(test_data_1), Base_sr._adata_format(test_data_2))
#         return train_dataset, test_dataset
    

# def data_prepare( data_path: str, train_split_path: str, test_split_path: str, key_1: str, key_2: str):

#     eval_data_obj = eval_data(data_path, train_split_path, test_split_path)
#     eval_data_obj.load_pair_data(key_1, key_2)
#     eval_data_obj.data_qc()
#     eval_data_obj.data_normalize()
#     train_dataset, test_dataset = eval_data_obj.get_train_test_dataset()
#     return train_dataset, test_dataset 

#*************************************************************************************************************************************************************
#*************************************************************************************************************************************************************
#*************************************************************************************************************************************************************
#*************************************************************************************************************************************************************


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

    


