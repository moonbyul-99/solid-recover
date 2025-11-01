import numpy as  np
import pandas as pd 
import scanpy as sc
import matplotlib.pyplot as plt 
import os 
import json
import sys 
sys.path.append('../../src')
from metrics import *

# def modified_get_latent(model):
#     '''
#     ori cobolt get_latent has a bug, when a dataloader return a batch with a single sample
#     '''



def eval(eval_dir):

    '''
    plot loss curve
    '''
    loss_path = os.path.join(eval_dir, 'loss.json')
    with open(loss_path,'r') as f:
        loss = json.load(f)

    loss = loss['loss']
    plt.plot(loss)
    plt.title(f'cobolt training loss')
    plt.savefig(os.path.join(eval_dir, 'loss.png'), bbox_inches='tight')

    '''
    match evaluation
    '''
    data_path = os.path.join(eval_dir, 'model_embed.h5ad')
    scdata = sc.read_h5ad(data_path)

    N = int(scdata.shape[0]/2)
    rna_embed = scdata.obsm['X_embed'][:N,:]
    atac_embed = scdata.obsm['X_embed'][N:,:]



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
    
    sc.pp.neighbors(scdata, use_rep='X_embed')
    sc.tl.umap(scdata)
    sc.pl.umap(scdata, color=['batch'], show = False)
    plt.savefig(os.path.join(eval_dir, 'embedding_umap.png'), bbox_inches='tight')  
    print('COBOLT eval OVER')
    return None 

if __name__ == '__main__':
    # eval_dir = '/home/rsun@ZHANGroup.local/solid-recover/compare_method/cobolt/case_2'
    # eval_dir = '/home/rsun@ZHANGroup.local/solid-recover/compare_method/cobolt/case_3'
    # eval_dir = '/home/rsun@ZHANGroup.local/solid-recover/compare_method/cobolt/case_4'
    # eval_dir = '/home/rsun@ZHANGroup.local/solid-recover/compare_method/cobolt/case_5'
    # eval_dir = '/home/rsun@ZHANGroup.local/solid-recover/compare_method/cobolt/case_6'
    # eval_dir = '/home/rsun@ZHANGroup.local/solid-recover/compare_method/cobolt/case_8'
    eval_dir = '/home/rsun@ZHANGroup.local/solid-recover/compare_method/cobolt/case_9'
    eval(eval_dir)  