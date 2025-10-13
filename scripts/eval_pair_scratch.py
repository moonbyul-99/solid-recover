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

def main():
    """Main function to run the training process from a config file."""
    parser = argparse.ArgumentParser(description='Train SR Model with Config')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--eval_dir', type = str, required=True, help='Path to the model directory need to be evaluated')
    parser.add_argument('--eval_result_dir', type = str, required=True, help='Path to the directory to save the evaluation results')
    args = parser.parse_args()

    # Load and print the config.
    config = load_config(args.config)
    print("Loaded config:")
    print(yaml.dump(config, default_flow_style=False))

    # === Data ===
    data_cfg = config['data']
    train_dataset, test_dataset = data_prepare(
        data_cfg['data_path'],
        data_cfg['train_split_path'],
        data_cfg['test_split_path'],
        data_cfg['key_1'],
        data_cfg['key_2'])
    
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
        #clip_temperature=model_cfg['clip_temperature'])
    
    # === Setup ===
    pair_model.set_dataset(train_dataset, test_dataset)
    pair_model.set_dataloader(batch_size=data_cfg['batch_size'])
    pair_model.set_loss(vae_beta_1= 1.0, vae_beta_2 = 1.0, clip_weight =  40, cross_recon_1 = 0.4,cross_recon_2 = 0.4,temperature = 0.07)

    # === eval each ckpt in eval_dir ===
    eval_res_dir = args.eval_result_dir
    os.makedirs(eval_res_dir, exist_ok=True)


    record = {}
    eval_dir = args.eval_dir
    for ckpt_path in tqdm(os.listdir(eval_dir)):
        tmp = ckpt_path.split('.')[0]
        steps = int(tmp.split('_')[-1])

        model_path = os.path.join(eval_dir, ckpt_path)
        pair_model.init_model(model_path)

        model = pair_model.model 
        model.to('cpu')
        model.eval()

        x1 = pair_model.test_dataset.omic_1.to('cpu')
        x2 = pair_model.test_dataset.omic_2.to('cpu')

        z, z_mu, z_logvar, z_embed = model.model_1.get_embedding(x1)
        y,y_mu, y_logvar, y_embed  = model.model_2.get_embedding(x2)

        N = z_embed.shape[0]
        z_mu = z_mu.detach().cpu().numpy()
        z_embed = z_embed.detach().cpu().numpy()

        y_mu = y_mu.detach().cpu().numpy()
        y_embed = y_embed.detach().cpu().numpy()

        ## perform evaluation
        res = {}
        ### calculate top_k hit rate
        tmp = []
        for i in [1,5,10,15,20,30,50,100]:
            res[f'top_{i}_hit'] = calculate_hit_rate(z_mu, y_mu, i, metric = 'cosine')

        ### calculate metric score
        acc, ms, fs = matching_metrics(x=z_mu, y = y_mu, metric='cosine')
        res['acc'] = acc 
        res['mathscore'] = ms 
        res['foscttm'] = fs
        print(f'{steps}:' + '========='*10)
        print(res)
        record[steps] = res 
    
        ### perform UMAP plot
        mu = np.concatenate([z_mu, y_mu], axis = 0)
        embed = np.concatenate([z_embed, y_embed], axis = 0)
        adata = ad.AnnData(X = np.random.rand(2*N,10))
        adata.obs.loc[:,'batch'] = ['rna']*N + ['atac']*N
        adata.obsm['mu'] = mu 
        adata.obsm['embed'] = embed 

        sc.pp.neighbors(adata, use_rep='mu')
        sc.tl.umap(adata, min_dist = 0.1)
        sc.pl.umap(adata, color='batch', show = False)
        plt.savefig(os.path.join(eval_res_dir, f'{steps}_umap.png'), bbox_inches='tight')

    df= pd.DataFrame(record)
    df = df.T
    df.sort_index(inplace=True)
    df.to_csv(os.path.join(eval_res_dir, 'eval_result_1.csv'))
    print('✅ TRAINING OVER')

if __name__ == '__main__':
    main()

