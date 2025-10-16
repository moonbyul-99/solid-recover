import numpy as  np 
import pandas as pd
import scanpy as sc 
import muon as mu  
import os
import sys 
sys.path.append('../../src')
from metrics import *
import scvi 

def match_eval(eval_res_dir):
    adata = sc.read_h5ad(os.path.join(eval_res_dir,'adata_mvi.h5ad'))

    rna_idx = np.logical_and(adata.obs.modality == 'expression',adata.obs.split == 'test')
    atac_idx = np.logical_and(adata.obs.modality == 'accessibility',adata.obs.split == 'test')

    rna_embed = adata.obsm['X_total_joint'][rna_idx,:]
    atac_embed = adata.obsm['X_total_joint'][atac_idx,:]
    print(rna_embed.shape, atac_embed.shape)


    '''check the cell barcode is consistent'''
    rna_cb = adata.obs.index.values[rna_idx]
    atac_cb = adata.obs.index.values[atac_idx] 

    for i in range(len(rna_cb)):
        x = rna_cb[i].split('rna_')[-1]
        x = x.split('_expression')[0] 

        y = atac_cb[i].split('atac_')[-1]
        y = y.split('_accessibility')[0]

        assert x== y

    '''calculate the matching index'''

    ## perform evaluation
    res = {}
    ### calculate top_k hit rate
    for i in [1,5,10,15,20,30,50,100]:
        res[f'top_{i}_hit'] = calculate_hit_rate(rna_embed,atac_embed, i, metric = 'euclidean')

    ### calculate metric score
    acc, ms, fs = matching_metrics(x=rna_embed, y = atac_embed, metric='euclidean')
    res['acc'] = acc 
    res['mathscore'] = ms 
    res['foscttm'] = fs
    with open(os.path.join(eval_res_dir, 'match_metric.json'), 'w') as f:
        json.dump(res, f)

    ### perform UMAP plot
    N = rna_embed.shape[0]
    mu = np.concatenate([rna_embed, atac_embed], axis = 0)
    adata = ad.AnnData(X = np.random.rand(2*N,10))
    adata.obs.loc[:,'batch'] = ['rna']*N + ['atac']*N
    adata.obsm['mu'] = mu 

    sc.pp.neighbors(adata, use_rep='mu')
    sc.tl.umap(adata, min_dist = 0.5)
    sc.pl.umap(adata, color='batch', show = False)
    plt.savefig(os.path.join(eval_res_dir, f'embedding_umap.png'), bbox_inches='tight')
    print(f'MULTIVI match Evaluation OVER')

def rna_pred_eval(eval_res_dir):
    
    adata = sc.read_h5ad(os.path.join(eval_res_dir,'adata_mvi.h5ad'))
    rna_idx = np.logical_and(adata.obs.modality == 'expression',adata.obs.split == 'test')
    atac_idx = np.logical_and(adata.obs.modality == 'accessibility',adata.obs.split == 'test')
    scvi.model.MULTIVI.setup_anndata(adata, batch_key="modality")

    model = scvi.model.MULTIVI.load(eval_res_dir, adata=adata)

    ### predict rna using atac### 
    imputed_expression = model.get_normalized_expression()
    print(imputed_expression.shape)

    '''atac predict rna evaluation'''

    atac2rna = imputed_expression.loc[atac_idx]
    print(atac2rna.shape)

    rna_var_idx = adata.var.modality == 'Gene Expression'
    rna_raw = adata[rna_idx, rna_var_idx].X.toarray()
    rna_var_names = atac2rna.columns
    print(rna_raw.shape)


    '''check the rna var name is consistent '''
    assert (adata.var.index[rna_var_idx] == atac2rna.columns).all()

    '''check the rna cell barcode is consistent'''
    rna_cb = adata.obs.index.values[rna_idx]
    atac_cb = atac2rna.index.values

    for i in range(len(rna_cb)):
        x = rna_cb[i].split('rna_')[-1]
        x = x.split('_expression')[0] 

        y = atac_cb[i].split('atac_')[-1]
        y = y.split('_accessibility')[0]

        assert x== y

    '''evaluation correlation between raw rna count and imputed rna count'''
    atac2rna = atac2rna.values    
    rna_corrs = pearson_corr_columns_scipy(rna_raw, atac2rna)
    res = {}
    for i,values in enumerate(rna_corrs):
        res[rna_var_names[i]] = values 
    with open(os.path.join(eval_res_dir, 'rna_pred_corr.json'), 'w') as f:
        json.dump(res, f)
    print(f'MULTIVI rna prediction Evaluation OVER')

def atac_pred_eval(eval_res_dir):
    adata = sc.read_h5ad(os.path.join(eval_res_dir,'adata_mvi.h5ad'))
    rna_idx = np.logical_and(adata.obs.modality == 'expression',adata.obs.split == 'test')
    atac_idx = np.logical_and(adata.obs.modality == 'accessibility',adata.obs.split == 'test')
    scvi.model.MULTIVI.setup_anndata(adata, batch_key="modality")

    model = scvi.model.MULTIVI.load(eval_res_dir, adata=adata)

    ### predict atac using rna###
    imputed_expression = model.get_accessibility_estimates()
    print(imputed_expression.shape)

    '''rna predict atac evaluation'''
    rna2atac = imputed_expression.loc[rna_idx]
    print(rna2atac.shape)

    atac2rna = imputed_expression.loc[atac_idx]
    print(atac2rna.shape)

    atac_var_idx = adata.var.modality == 'Peaks'
    atac_raw = adata[atac_idx, atac_var_idx].X.toarray()
    atac_var_names = rna2atac.columns
    rna2atac = rna2atac.loc[:,atac_var_names]
    print(f'ori atac shape {atac_raw.shape}, pred atac based on rna {rna2atac.shape}')


    '''check the rna cell barcode is consistent'''
    rna_cb = rna2atac.index
    atac_cb = adata.obs.index.values[atac_idx]

    for i in range(len(rna_cb)):
        x = rna_cb[i].split('rna_')[-1]
        x = x.split('_expression')[0] 

        y = atac_cb[i].split('atac_')[-1]
        y = y.split('_accessibility')[0]

        assert x== y

    '''evaluation correlation between raw rna count and imputed rna count'''
    rna2atac = rna2atac.values    
    atac_corrs = pearson_corr_columns_scipy(atac_raw, rna2atac)
    res = {}
    for i,values in enumerate(atac_corrs):
        res[atac_var_names[i]] = values  
    with open(os.path.join(eval_res_dir, 'atac_pred_corr.json'), 'w') as f:
        json.dump(res, f)
    print(f'MULTIVI atac prediction Evaluation OVER')

def mvi_eval(eval_res_dir):
    print('evaluate match task')
    match_eval(eval_res_dir) 
    print('evaluate rna prediction task')
    rna_pred_eval(eval_res_dir)
    print('evaluate atac prediction task')
    atac_pred_eval(eval_res_dir)
    print('MULTIVI Evaluation OVER')

if __name__ == '__main__':
    ## modify eval_res_dir"
    # eval_res_dir = '/home/rsun@ZHANGroup.local/solid-recover/compare_method/multivi/case_1'
    # eval_res_dir = '/home/rsun@ZHANGroup.local/solid-recover/compare_method/multivi/case_2'
    # eval_res_dir = '/home/rsun@ZHANGroup.local/solid-recover/compare_method/multivi/case_3'
    # eval_res_dir = '/home/rsun@ZHANGroup.local/solid-recover/compare_method/multivi/case_4'
    eval_res_dir = '/home/rsun@ZHANGroup.local/solid-recover/compare_method/multivi/case_4_count'
    mvi_eval(eval_res_dir)