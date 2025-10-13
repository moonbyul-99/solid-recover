import sys 
sys.path.append('src')
from sr_model import *
from sr_dataset import *
from load_eval_data import *
from metrics import calculate_hit_rate, matching_metrics

import os
import numpy as np 
import pandas as pd 
from typing import List, Dict, Union, Any 
import muon as mu 
import scanpy as sc
import yaml 

import argparse
import os
from load_eval_data import *

import json
import matplotlib.pyplot as plt 
import seaborn as sns 
import anndata as ad
from tqdm import tqdm


def load_config(config_path):
    """Loads and parses a YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config 

def pred_pipe(config_path, ckpt_path, eval_dir):
    os.makedirs(eval_dir, exist_ok=True)

    
    config = load_config(config_path)
    # === Data ===
    data_cfg = config['data']
    train_dataset, test_dataset = data_prepare(
        data_cfg['data_path'],
        data_cfg['train_split_path'],
        data_cfg['test_split_path'],
        data_cfg['key_1'],
        data_cfg['key_2']
    )

    # === Model ===
    model_cfg = config['model']

    # Dynamically determine feature_num_1 and feature_num_2
    # This assumes the first sample in the dataset is representative.
    feature_num_1 = len(train_dataset[0]['omic_1'])
    feature_num_2 = len(train_dataset[0]['omic_2'])

    pair_model = pair_sr_scratch(
        feature_num_1=feature_num_1,
        feature_num_2=feature_num_2,
        hidden_params_1=model_cfg['hidden_params_1'],
        hidden_params_2=model_cfg['hidden_params_2'],
        embed_dim=model_cfg['embed_dim'],
        use_rmsnorm=model_cfg['use_rmsnorm'],
        use_residual=model_cfg['use_residual'],
        dropout_p=model_cfg['dropout_p'],)
    
    #=== Setup ===
    pair_model.set_dataset(train_dataset, test_dataset)
    pair_model.set_dataloader(batch_size=data_cfg['batch_size'])
    pair_model.set_loss(vae_beta_1= 1.0, vae_beta_2 = 1.0, clip_weight =  40, cross_recon_1 = 0.4,cross_recon_2 = 0.4,temperature = 0.07)


    pair_model.init_model(ckpt_path)
    model = pair_model.model

    #===prediction===
    model = pair_model.model 
    model.to('cpu')
    model.eval()

    x1 = pair_model.test_dataset.omic_1.to('cpu')
    x2 = pair_model.test_dataset.omic_2.to('cpu')
    print(x1.shape, x2.shape)

    outputs = model(x1,x2)

    rna2atac = outputs['x2_c_recon'].detach().numpy()
    atac2rna = outputs['x1_c_recon'].detach().numpy()
    print(rna2atac.shape, atac2rna.shape)


    #=== save predict result =====
    X = np.concatenate([x1, atac2rna], axis = 0)
    adata = ad.AnnData(X)
    adata.obs['label'] = ['ori_rna']*x1.shape[0] + ['pred_rna']*x1.shape[0]
    adata.write_h5ad(os.path.join(eval_dir, 'rna_test.h5ad'))

    X = np.concatenate([x2, rna2atac], axis = 0)
    adata = ad.AnnData(X)
    adata.obs['label'] = ['ori_atac']*x1.shape[0] + ['pred_atac']*x1.shape[0]
    adata.write_h5ad(os.path.join(eval_dir, 'atac_test.h5ad'))

    print('Pred pipe over')



from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def pearson_corr_columns_scipy(A, B):
    p = A.shape[1]
    corrs = np.empty(p)
    for i in range(p):
        corrs[i], _ = pearsonr(A[:, i], B[:, i])
    return corrs
def eval_pipe(eval_dir, type = 'rna'):
    adata = ad.read(os.path.join(eval_dir, f'{type}_test.h5ad'))   

    ## visualization
    print('Perform visualization')
    sc.tl.pca(adata)
    sc.pl.pca(adata, color = 'label', return_fig = False)
    plt.savefig(os.path.join(eval_dir, f'{type}_pca_vis.png'), bbox_inches='tight')

    sc.pp.neighbors(adata)
    sc.tl.umap(adata, min_dist = 0.5)
    sc.pl.umap(adata, color = 'label', return_fig = False)
    plt.savefig(os.path.join(eval_dir, f'{type}_umap_vis.png'), bbox_inches='tight')

    ## perform classification evaluation 
    print('Perform classification evaluation')
    le = LabelEncoder()
    y = le.fit_transform(adata.obs['label'].values)

    lg = LogisticRegression(max_iter = 1000)
    knn = KNeighborsClassifier(n_neighbors=5) 

    train_id, test_id = train_test_split(np.arange(adata.shape[0]), test_size=0.2, random_state=42) 
    train_x = adata[train_id].X
    train_y = y[train_id]

    test_x = adata[test_id].X
    test_y = y[test_id] 
    print('=======lg model training...')
    lg.fit(train_x, train_y)
    pred_prob = lg.predict_proba(test_x)
    auc = roc_auc_score(test_y, pred_prob[:,1])
    acc = accuracy_score(test_y, lg.predict(test_x))
    lg_res = {'auc':auc, 'acc':acc}
    with open(os.path.join(eval_dir, f'{type}_lg_res.json'), 'w') as f:
        json.dump(lg_res, f)

    print('======knn model training...')
    knn.fit(train_x, train_y)
    pred_prob = knn.predict_proba(test_x)
    auc = roc_auc_score(test_y, pred_prob[:,1])
    acc = accuracy_score(test_y, knn.predict(test_x))
    # print('knn auc:', auc, 'knn acc:', acc)
    knn_res = {'auc':auc, 'acc':acc}
    with open(os.path.join(eval_dir, f'{type}_knn_res.json'), 'w') as f:
        json.dump(knn_res, f)

    ## feature corr evaluation 
    print('Perform feature correlation evaluation')
    N = int(adata.shape[0]/2)
    A = adata[:N,:].X
    B = adata[N:,:].X
    corrs = pearson_corr_columns_scipy(A,B)
    np.save(os.path.join(eval_dir, f'{type}_corr.npy'), corrs)

    print(f'{type} evaluation over')

def main():
    parser = argparse.ArgumentParser(description='prediction evaluation of sr model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--eval_dir', type=str, required=True, help='Path to eval dir')

    args = parser.parse_args()

    print('Load model and perform cross prediction')
    pred_pipe(args.config, args.ckpt_path, args.eval_dir)
    print('Perform evaluation of rna')
    eval_pipe(args.eval_dir, type = 'rna')
    print('Perform evaluation of atac')
    eval_pipe(args.eval_dir, type = 'atac')
    print('OVER')

if __name__ == '__main__':
    main()



        