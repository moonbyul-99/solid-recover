# script for rna pretraining 
import sys
import os
import argparse
import yaml
sys.path.append('src')
from sr_model import single_sr
from datasets import load_from_disk

def load_config(config_path):
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='Train SR Model with Config')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print("Loaded config:")
    print(yaml.dump(config, default_flow_style=False))

    # === Data ===
    data_cfg = config['data']
    dataset = load_from_disk(data_cfg['dataset_path'])
    dataset_dic = dataset.train_test_split(
        test_size=data_cfg['test_size'],
        seed=data_cfg['seed']
    )
    train_dataset, test_dataset = dataset_dic['train'], dataset_dic['test']

    # Get feature_num from first sample
    feature_num = dataset[0]['feature'].shape[0]

    # === Model ===
    model_cfg = config['model']
    rna_model = single_sr(
        feature_num=feature_num,
        hidden_params=model_cfg['hidden_params'],
        embed_dim=model_cfg['embed_dim'],
        use_rmsnorm=model_cfg['use_rmsnorm'],
        use_residual=model_cfg['use_residual'],
        dropout_p=model_cfg['dropout_p'],
        vae_model=model_cfg['vae_model']
    )

    # === Setup ===
    rna_model.set_dataset(train_dataset, test_dataset)
    rna_model.set_dataloader(batch_size=data_cfg['batch_size'])

    loss_cfg = config.get('loss', {})
    rna_model.set_loss(beta=loss_cfg.get('beta', 1.0))

    opt_cfg = config['optimizer']
    rna_model.set_optimizer(
        lr=float(opt_cfg['lr']),
        warmup_steps=opt_cfg['warmup_steps'],
        steady_1_steps=opt_cfg['steady_1_steps'],
        cosine_anneal_steps=opt_cfg['cosine_anneal_steps'],
        min_lr=float(opt_cfg['min_lr'])
    )

    train_cfg = config['training']
    rna_model.set_project(train_cfg['project_dir'])

    # === Train ===
    rna_model.train_model(
        train_steps=train_cfg['train_steps'],
        eval_points=train_cfg['eval_points'],
        save_points=train_cfg['save_points'],
        device=train_cfg.get('device', 'cuda')
    )

    print('✅ TRAINING OVER')

if __name__ == '__main__':
    main()