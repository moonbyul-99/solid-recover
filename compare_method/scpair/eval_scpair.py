import scanpy as sc 
import numpy as np
import pandas as pd
import anndata as ad
import os 
import matplotlib.pyplot as plt
import json 
import sys 
sys.path.append('../../src')
from metrics import * 


def eval_pipe(eval_dir):
    '''
    load adata
    '''
    data_path = os.path.join(eval_dir, 'scdata.h5ad')
    scdata = sc.read_h5ad(data_path) 

    '''
    perform match eval 
    '''

    rna_embed = scdata.obsm['gene']
    atac_embed = scdata.obsm['p2g']

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
    with open(os.path.join(eval_dir, 'match_metric.json'), 'w') as f:
        json.dump(res, f)

    ### UMAP visualization  ###
    N = scdata.shape[0]
    adata = ad.AnnData(X = np.random.rand(2*N,10))
    adata.obs.loc[:,'batch'] = ['rna']*N + ['atac']*N
    mu = np.concatenate([rna_embed, atac_embed], axis = 0)
    adata.obsm['mu'] = mu   
    sc.pp.neighbors(adata, use_rep='mu')    
    sc.tl.umap(adata, min_dist = 0.5)
    sc.pl.umap(adata, color='batch')
    plt.savefig(os.path.join(eval_dir, 'embedding_umap.png'), bbox_inches='tight')
    print('scpair evaluation OVER')

if __name__ == '__main__':
    #eval_dir = 'case_1'
    #eval_dir = 'case_3'
    # eval_dir = 'case_4'
    # eval_dir = 'case_2'
    # eval_dir = 'case_5'
    # eval_dir = 'case_6'
    # eval_dir = 'case_8'
    eval_dir = 'case_9'
    eval_pipe(eval_dir)