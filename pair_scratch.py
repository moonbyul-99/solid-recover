import sys 
sys.path.append('src')
from sr_model import *
from sr_dataset import *
from load_eval_data import *
import os
import numpy as np 
import pandas as pd 
from typing import List, Dict, Union, Any 
import anndata
import sys 
import yaml 
import argparse
import shutil
import copy
import warnings 
# --- 全局设置 ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=anndata.ImplicitModificationWarning)

def load_config(config_path):
    """Loads and parses a YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """Main function to run the training process from a config file."""
    parser = argparse.ArgumentParser(description='Train SR Model with Config')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--to_gpu', type=str, default='False', help='Whether to move dataset into GPU')
    args = parser.parse_args()

    # Load and print the config.
    config = load_config(args.config)
    # original_config = copy.deepcopy(config)
    print("Loaded config:")
    # print(yaml.dump(config, default_flow_style=False))

    # === prepare dataset ===
    data_cfg = config['data']
    train_data_path = data_cfg['train_data_path']
    test_data_path = data_cfg['test_data_path']
    key_1, key_2 = data_cfg['key_1'], data_cfg['key_2']
    if args.to_gpu == 'False':
        to_gpu = False
    elif args.to_gpu == 'True':
        to_gpu = True
    else:
        raise ValueError('Invalid value for to_gpu: {}'.format(args.to_gpu))
    train_dataset, test_dataset = data_prepare(train_data_path, test_data_path, key_1, key_2, to_gpu)
    
    # === Model ===
    model_cfg = config['model']
    
    ## assert that the input feature number matches the feature number in the first sample.
    feature_num_1 = model_cfg['feature_num_1']
    feature_num_2 = model_cfg['feature_num_2']
    assert feature_num_1 == len(train_dataset[0]['omic_1'])
    assert feature_num_2 == len(train_dataset[0]['omic_2'])

    pair_model = pair_sr_scratch(
        feature_num_1=feature_num_1,
        feature_num_2=feature_num_2,
        hidden_params_1=model_cfg['hidden_params_1'],
        hidden_params_2=model_cfg['hidden_params_2'],
        embed_dim=model_cfg['embed_dim'],
        use_rmsnorm=model_cfg['use_rmsnorm'],
        use_residual=model_cfg['use_residual'],
        dropout_p=model_cfg['dropout_p'],)

    # === Setup ===
    pair_model.set_dataset(train_dataset, test_dataset)
    pair_model.set_dataloader(batch_size=data_cfg['batch_size'])

    # === Loss ===
    loss_cfg = config['loss']
    pair_model.set_loss(
        vae_beta_1=loss_cfg['vae_beta_1'],
        vae_beta_2=loss_cfg['vae_beta_2'],
        clip_weight=float(loss_cfg['clip_weight']),
        cross_recon_1=float(loss_cfg['cross_recon_1']),
        cross_recon_2=float(loss_cfg['cross_recon_2']),
        temperature=float(loss_cfg['temperature']),
        trainable_clip_temperature=loss_cfg['trainable_clip_temperature'],
        use_weight=loss_cfg['use_weight'],
        top_k_ratio=loss_cfg['top_k_ratio'],
        bottom_k_ratio=loss_cfg['bottom_k_ratio'],
        weight_top=loss_cfg['weight_top'],
        weight_bottom=loss_cfg['weight_bottom']
    )
    
    # === Optimizer ===
    opt_cfg = config['optimizer']
    pair_model.set_optimizer(
        lr=float(opt_cfg['lr']),
        warmup_steps=opt_cfg['warmup_steps'],
        steady_1_steps=opt_cfg['steady_1_steps'],
        cosine_anneal_steps=opt_cfg['cosine_anneal_steps'],
        min_lr=float(opt_cfg['min_lr'])
    )

    train_cfg = config['training']
    pair_model.set_project(train_cfg['project_dir'])

    # === Copy config to project dir ===
    import shutil
    config_copy_path = os.path.join(pair_model.project_dir, 'config.yaml')
    shutil.copyfile(args.config, config_copy_path)
    # === Save the exact config used in this run ===
    # config_save_path = os.path.join(pair_model.project_dir, 'config.yaml')
    # # os.makedirs(pair_model.project_dir, exist_ok=True)  # 确保目录存在
    # with open(config_save_path, 'w') as f:
    #     yaml.safe_dump(
    #         original_config,
    #         f,
    #         default_flow_style=True,
    #         indent=2,
    #         width=1000,
    #         allow_unicode=True,
    #         sort_keys=False
    #     )


    # === Train ===
    pair_model.train_model(
        train_steps=train_cfg['train_steps'],
        eval_points=train_cfg['eval_points'],
        save_points=train_cfg['save_points'],
        device=train_cfg['device']
    )

    print('✅ TRAINING OVER')

if __name__ == '__main__':
    main()