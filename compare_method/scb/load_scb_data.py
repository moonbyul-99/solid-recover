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
import muon as mu 
from mudata import MuData

import anndata as ad
import pandas as pd
import re

def sort_var_by_chrom_and_start(adata):
    """
    对 adata.var 按染色体顺序（chr1, chr2, ..., chrX, chrY）和 start 位置升序排序。
    原地修改 adata。
    """
    # 1. 解析 index 为 (chrom, start, end)
    def parse_peak(peak_str):
        # 支持 'chr1:1000-2000' 或 '1:1000-2000' 等格式
        match = re.match(r'^(?:chr)?([0-9XYM]+):(\d+)-(\d+)$', peak_str, re.IGNORECASE)
        if not match:
            raise ValueError(f"无法解析 peak 名称: {peak_str}")
        chrom, start, end = match.groups()
        return chrom, int(start), int(end)
    
    # 2. 创建排序用的 DataFrame
    parsed = [parse_peak(idx) for idx in adata.var.index]
    sort_df = pd.DataFrame(parsed, columns=['chrom', 'start', 'end'], index=adata.var.index)
    
    # 3. 定义染色体排序顺序
    chrom_order = {}
    for i in range(1, 23):
        chrom_order[str(i)] = i
    chrom_order['X'] = 23
    chrom_order['Y'] = 24
    chrom_order['M'] = 25  # 可选：线粒体
    
    # 处理可能的小写 'x', 'y', 'm'
    sort_df['chrom'] = sort_df['chrom'].str.upper()
    
    # 转换 chrom 为排序数值，未知染色体放最后
    sort_df['chrom_rank'] = sort_df['chrom'].map(chrom_order).fillna(999)
    
    # 4. 排序：先按 chrom_rank，再按 start
    sort_df = sort_df.sort_values(['chrom_rank', 'start'])
    
    # 5. 重新索引 adata
    adata._inplace_subset_var(sort_df.index)



def load_data(train_data_path, test_data_path):
    '''
    Load the train and test data, concat to mdata
    '''

    train_mdata = mu.read_h5mu(train_data_path)
    test_mdata = mu.read_h5mu(test_data_path)


    '''
    concat to mdata
    '''
    res = {}
    for key in ['rna_count', 'peak_count']:
        adata = ad.concat([train_mdata.mod[key], test_mdata.mod[key]])
        adata.obs.loc[:,'split'] = ['train']*train_mdata.mod[key].shape[0] + ['test']*test_mdata.mod[key].shape[0]
        adata.var = train_mdata.mod[key].var.copy()
        res[key] = adata 
    mdata = MuData(res)

    '''
    save only rna and atac data, filter rna var with NA interval
    '''
    rna = mdata['rna_count']
    atac = mdata['peak_count']

    '''filter rna var with NA interval'''
    drop_id = rna.var.interval == 'NA'
    rna = rna[:, ~drop_id]

    #mdata = MuData({'rna_count': rna, 'peak_count': atac})
    
    '''
    generate the train id and test id using int index
    '''
    IDX = np.arange(rna.shape[0])
    train_id = IDX[rna.obs.split == 'train']
    test_id = IDX[rna.obs.split == 'test']

    # drop peak not in chr 
    peak_sel = []
    for ele in atac.var.index.values:
        if 'chr' in ele:
            peak_sel.append(True)
        else:
            peak_sel.append(False)
    atac = atac[:,peak_sel]
   
    ## perform scb preprocessing 

    print('perform scb RNA preprocessing')
    RNA_data = RNA_data_preprocessing(
        rna,
        normalize_total=True,
        log1p=True,
        use_hvg=False,
        n_top_genes=8000,
        save_data=False,
        file_path=None,
        logging_path=None
        )
    
    print('perform scb ATAC preprocessing')

    ATAC_data = ATAC_data_preprocessing(
        atac,
        binary_data=True,
        filter_features=False,
        fpeaks=0.005,
        tfidf=False,  #TIME CONSUMING, VERY LOW EFFICENCY IMPLEMENTATION
        normalize=True,
        save_data=False,
        file_path=None,
        logging_path=None
    )[0]

    ## scb chrom preprocess 
    print('Additional SCB atac preprocessing')

    '''peak sort for scb only'''
    sort_var_by_chrom_and_start(ATAC_data)

    chrom = []
    for ele in ATAC_data.var.index:
        a = ele.split(':')[0]
        if 'chr' in a:
            chrom.append(a)
        else:
            chrom.append('NA')
    ATAC_data.var['chrom'] = chrom

    chrom_list = []
    last_one = ''
    for i in range(len(ATAC_data.var.chrom)):
        temp = ATAC_data.var.chrom[i]
        if temp[0 : 3] == 'chr':
            if not temp == last_one:
                chrom_list.append(1)
                last_one = temp
            else:
                chrom_list[-1] += 1
        else:
            chrom_list[-1] += 1

    print(chrom_list, end="")

    return RNA_data, ATAC_data, test_id, train_id, chrom_list