import numpy as np 
import pandas as pd
import anndata as ad 
import scanpy as sc
import matplotlib.pyplot as plt
import os 
import seaborn as sns 
from utils import *
from match_compare_lineplot import *
from visualization import * 
from tqdm import tqdm
import sys
sys.path.append('../src')
from metrics import *
import muon as mu 
sys.path.append('../')
from pair_eval import *

'''
获取某个实验下的全部test数据的双模态embedding, 保存在一个统一的数据下
'''

def get_atac_embed(case_id, scdata):
    '''
    case_id: 某个实验id, eg. case_8 
    scdata: 这个实验id下的test rna count 数据
    '''

    atac_dic = {}
    #case_id = 'case_8'
    '''cobolt embed'''
    adata = sc.read_h5ad(f'/home/rsun@ZHANGroup.local/solid-recover/compare_method/cobolt/{case_id}/cobolt_latent.h5ad')
    sel_boolidx = []
    sel_idx = []
    for ele in adata.obs.index:
        if 'test_atac' in ele:
            sel_idx.append(ele.split('~')[-1])
            sel_boolidx.append(True)
        else:
            sel_boolidx.append(False)

    adata = adata[sel_boolidx]
    adata.obs.index = sel_idx 
    adata = adata[scdata.obs.index]
    atac_dic['cobolt'] = adata.X.copy()

    '''mvi'''
    mvi_embed = sc.read_h5ad(f'/home/rsun@ZHANGroup.local/solid-recover/compare_method/multivi/{case_id}/adata_mvi.h5ad')
    mvi_idx = np.logical_and(mvi_embed.obs.split.values == 'test' , mvi_embed.obs.modality.values == 'accessibility')
    mvi_embed = mvi_embed.obsm['X_total_atac'][mvi_idx,:]
    atac_dic['mvi'] = mvi_embed

    '''scb'''
    scb_embed = sc.read_h5ad(f'/home/rsun@ZHANGroup.local/solid-recover/compare_method/scb/{case_id}/sc_test.h5ad')
    atac_dic['scb'] = scb_embed.obsm['A2R'].copy()


    '''scpair'''
    scpair_embed = sc.read_h5ad(f'/home/rsun@ZHANGroup.local/solid-recover/compare_method/scpair/{case_id}/scdata.h5ad')
    atac_dic['scpair'] = scpair_embed.obsm['p2g'].copy()
    return atac_dic 

def get_rna_embed(case_id, scdata):

    rna_dic = {}

    '''cobolt embed'''
    adata = sc.read_h5ad(f'/home/rsun@ZHANGroup.local/solid-recover/compare_method/cobolt/{case_id}/cobolt_latent.h5ad')
    sel_boolidx = []
    sel_idx = []
    for ele in adata.obs.index:
        if 'test_rna' in ele:
            sel_idx.append(ele.split('~')[-1])
            sel_boolidx.append(True)
        else:
            sel_boolidx.append(False)

    adata = adata[sel_boolidx]
    adata.obs.index = sel_idx 
    adata = adata[scdata.obs.index]
    rna_dic['cobolt'] = adata.X.copy()

    '''mvi'''
    mvi_embed = sc.read_h5ad(f'/home/rsun@ZHANGroup.local/solid-recover/compare_method/multivi/{case_id}/adata_mvi.h5ad')
    mvi_idx = np.logical_and(mvi_embed.obs.split.values == 'test' , mvi_embed.obs.modality.values == 'expression')
    mvi_embed = mvi_embed.obsm['X_total_rna'][mvi_idx,:]
    #scdata.obsm['mvi_embed'] = mvi_embed 
    rna_dic['mvi'] = mvi_embed

    '''scb'''
    scb_embed = sc.read_h5ad(f'/home/rsun@ZHANGroup.local/solid-recover/compare_method/scb/{case_id}/sc_test.h5ad')
    #scdata.obsm['scb_embed'] = scb_embed.obsm['R2R'].copy()
    rna_dic['scb'] = scb_embed.obsm['R2R'].copy()


    '''scpair'''
    scpair_embed = sc.read_h5ad(f'/home/rsun@ZHANGroup.local/solid-recover/compare_method/scpair/{case_id}/scdata.h5ad')
    #scdata.obsm['scpair_embed'] = scpair_embed.obsm['gene'].copy()
    rna_dic['scpair'] = scpair_embed.obsm['gene'].copy()
    return rna_dic 

def fetch_sr_embed(sr_dir, scdata,  info, start_ckpt):
    '''
    获取某个试验下sr的不同ckpt的全部embedding
    sr_dir: 某个试验的sr保存目录
    start_ckpt: 从某个ckpt开始获取embedding
    info: 指明是weighted_sr 还是原始的 sr
    ''' 
    # sr_dir = '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case8_wc_new_20251102_0359'
    pair_model = prepare_evaluation(sr_dir)
    for ckpt_point in tqdm(range(start_ckpt, 10000, 500)):

        ckpt_path = f'{sr_dir}/models/ckpt_{ckpt_point}.pth'
        try:
            result = get_sr_embed(pair_model, ckpt_path, dataset = 'test')
            scdata.obsm[f'rna_{info}_{ckpt_point}'] = result['rna_mu']
            scdata.obsm[f'atac_{info}_{ckpt_point}'] = result['atac_mu']
        except: 
            print(f'{sr_dir} ckpt {ckpt_point} error')


    return scdata  

def fetch_sr_embed_single( sr_dir, scdata, ckpt_point, info):
    '''
    获取某个试验下sr的单个ckpt的全部embedding
    sr_dir: 某个试验的sr保存目录
    start_ckpt: 从某个ckpt开始获取embedding
    info: 指明是weighted_sr 还是原始的 sr
    ''' 
    # sr_dir = '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case
    pair_model = prepare_evaluation(sr_dir)

    ckpt_path = f'{sr_dir}/models/ckpt_{ckpt_point}.pth'
    result = get_sr_embed(pair_model, ckpt_path, dataset = 'test')
    scdata.obsm[f'rna_{info}_{ckpt_point}'] = result['rna_mu']
    scdata.obsm[f'atac_{info}_{ckpt_point}'] = result['atac_mu']

    return scdata  

def pipeline(case_id, sr_dir, wc_start_ckpt, sr_ori_dir, ori_ckpt, save_dir):
    '''
    sr_dir: 最终评测的sr保存目录 
    wc_start_ckpt: 从某个ckpt开始获取weighted_sr embedding 

    sr_ori_dir: 获取原始sr的保存目录
    ori_start_ckpt: 从某个ckpt开始获取原始sr embedding
    '''

    os.makedirs(save_dir, exist_ok = True) 

    '''step 1 读取test_count 数据，记录初始rna_embed 和 atac_embed'''
    print(f'load {case_id} data, get rna pca embedding and atac pca embedding ============================')
    sr_raw = mu.read_h5mu(f'/home/rsun@ZHANGroup.local/solid-recover/data/{case_id}/test_count.h5mu')

    scdata = sr_raw['rna_count']
    sc.pp.normalize_total(scdata, target_sum=1e4)
    sc.pp.log1p(scdata)
    sc.tl.pca(scdata)
    scdata.obsm['rna_raw'] = scdata.obsm['X_pca'].copy()

    adata = sr_raw['peak_count']
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata) 
    sc.tl.pca(adata)
    scdata.obsm['atac_raw'] = adata.obsm['X_pca'].copy() 

    '''step 2 获取对比方法embedding'''
    print(f'get {case_id} compare method embedding ============================')
    rna_dic = get_rna_embed(case_id, scdata)
    atac_dic = get_atac_embed(case_id, scdata)
    for key in rna_dic.keys():
        scdata.obsm[f'rna_{key}'] = rna_dic[key]
    for key in atac_dic.keys():
        scdata.obsm[f'atac_{key}'] = atac_dic[key] 

    scdata.write_h5ad(os.path.join(save_dir, 'test_data.h5ad')) 

    '''step 3 获取sr 策略的embedding'''
    print(f'get {case_id} sr embedding ============================')
    scdata = fetch_sr_embed(sr_dir, scdata, info = 'wc', start_ckpt=wc_start_ckpt, )
    scdata.write_h5ad(os.path.join(save_dir, 'test_data.h5ad')) 

    '''step 4 获取原始sr 策略的embedding'''
    print(f'get {case_id} ORI sr embedding ============================')
    scdata = fetch_sr_embed_single( sr_ori_dir, scdata = scdata, ckpt_point= ori_ckpt, info = 'ori')
    scdata.write_h5ad(os.path.join(save_dir, 'test_data.h5ad'))

if __name__ == '__main__':
    # case_id = 'case_8'
    # sr_dir = '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case8_wc_new_20251102_0359'
    # wc_start_ckpt = 2500
    # sr_ori_dir = '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case8_wc_noweight'
    # ori_ckpt = 4000
    # save_dir = 'case8_embedding'

    # case_id = 'case_9'
    # sr_dir = '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case9_wc_20251101_1439/'
    # wc_start_ckpt = 1000
    # sr_ori_dir = '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case9_oriclip/'
    # ori_ckpt = 1500
    # save_dir = 'case9_embedding'

    # case_id = 'case_11' 
    # sr_dir = '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case11_wc_20251104_1421/'
    # wc_start_ckpt = 1000
    # sr_ori_dir = '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case11_wc_oriclip'
    # ori_ckpt = 1500
    # save_dir = 'case11_embedding'

    case_id = 'case_12'
    sr_dir = '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case12_wc'
    wc_start_ckpt = 500 
    sr_ori_dir = '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case12_wc_oriclip'
    ori_ckpt = 2000
    save_dir = 'case12_embedding'

    pipeline(case_id, sr_dir, wc_start_ckpt, sr_ori_dir, ori_ckpt, save_dir) 


