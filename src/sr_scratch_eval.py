from sr_model import single_sr, paired_sr, phase_1_train, phase_2_train
from sr_net import multi_model
import numpy as np
import muon as mu 
import scanpy as sc
from dataset import single_data, omic_data
import yaml 
import torch 
from torch.utils.data import Dataset, DataLoader 
import torch.nn as nn 
import os 
import argparse

'''
Improve the scratch model performance,
this code is modified from sr_eval.py
'''

def load_data(eval_id):
    mdata = mu.read_h5mu('/home/rsun@ZHANGroup.local/sr_project/eval_data/merge_dataset/mdata.h5mu')
    
    rna = mdata['rna_count']
    gadata = mdata['ga_count'] 

    mutual_gene = np.load('/home/rsun@ZHANGroup.local/atac_pretrain/src/mutual_gene.npy', allow_pickle=True)
    gadata = gadata[:,mutual_gene]
    
    # lo1p transform 

    sc.pp.normalize_total(gadata, target_sum= 1e4)
    sc.pp.log1p(gadata)

    sc.pp.normalize_total(rna, target_sum= 1e4)
    sc.pp.log1p(rna)
    
    print(rna.shape, gadata.shape)

    path_1 = f'/home/rsun@ZHANGroup.local/sr_project/eval_data/data_split/train_id_{eval_id}.npy'
    path_2 = f'/home/rsun@ZHANGroup.local/sr_project/eval_data/data_split/test_id_{eval_id}.npy'
    #path_1 = f'/home/rsun@ZHANGroup.local/multi_pretrain/evaluation/sr_result/train_test_split/eval_{eval_id}_train_id.npy'
    #path_2 = f'/home/rsun@ZHANGroup.local/multi_pretrain/evaluation/sr_result/train_test_split/eval_{eval_id}_test_id.npy' 

    train_idx = np.load(path_1, allow_pickle=True)
    test_idx = np.load(path_2, allow_pickle=True) 

    rna_train, rna_test = rna[train_idx,:], rna[test_idx,:]
    gadata_train, gadata_test = gadata[train_idx,:], gadata[test_idx,:]

    '''
    rna data preprocess
    '''

    rna_train = rna_train.X.toarray().astype(np.float32)
    rna_test = rna_test.X.toarray().astype(np.float32)

    rna_train = torch.from_numpy(rna_train)
    rna_test = torch.from_numpy(rna_test)

    '''
    gadata preprocess
    '''

    gadata_train = gadata_train.X.toarray().astype(np.float32)
    gadata_test = gadata_test.X.toarray().astype(np.float32)

    gadata_train = torch.from_numpy(gadata_train)
    gadata_test = torch.from_numpy(gadata_test)

    '''
    set dataset
    '''

    train_dataset = omic_data(rna_train, gadata_train)
    test_dataset = omic_data(rna_test, gadata_test)

    return train_dataset, test_dataset

def initialize_model(model):
    ''' 
    Initialize the model randomly
    '''
    # Reset parameters of the model
    for param in model.parameters():
        if param.requires_grad:
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)
    print('Model initialized randomly')
    return model

def freeze_param_single(model):
    '''
    freeze the single sr model param
    '''
    for param in model.encoder_net.encoder_header.parameters():
        param.requires_grad = False
    for param in model.decoder_net.decoder_header.parameters():
        param.requires_grad = False    
    return model

def unfreeze_param_single(model):
    '''
    Unfreeze the single sr model param
    '''
    for param in model.encoder_net.encoder_header.parameters():
        param.requires_grad = True
    for param in model.decoder_net.decoder_header.parameters():
        param.requires_grad = True
    return model
    
def main(eval_id,
         params):
    '''
    get dataset
    '''

    train_dataset, test_dataset = load_data(eval_id)

    '''
    get dataloader
    '''

    batch_size = params['batch_size']

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle= True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle= True)

    print('Dataloader over')

    '''
    prepare model
    ''' 

    config_rna = params['rna_config']
    config_ga = params['ga_config']
    rna_checkpoint_path = params['rna_checkpoint_path']
    ga_checkpoint_path = params['ga_checkpoint_path']
    
    ''' prepare rna model '''
    with open(config_rna,'r') as f:
        config = yaml.safe_load(f)

    sr_rna = single_sr(config)
    checkpoint = torch.load(rna_checkpoint_path)
    sr_rna.model.load_state_dict(checkpoint['model_state_dict'])

    rna_model = sr_rna.model

    ''' prepare ga model '''
    with open(config_ga,'r') as f:
        config = yaml.safe_load(f)

    sr_ga = single_sr(config)
    checkpoint = torch.load(ga_checkpoint_path)
    sr_ga.model.load_state_dict(checkpoint['model_state_dict'])

    ga_model = sr_ga.model 

    '''
    initialize model
    '''
    rna_model = initialize_model(rna_model)
    ga_model = initialize_model(ga_model)


    rna_model = unfreeze_param_single(rna_model)
    ga_model = unfreeze_param_single(ga_model)
    paired_model = multi_model(rna_model, ga_model)#, temperature = 0.1, rna_weight = 1, ga_weight = 1)

    '''
    paired model training
    '''
    print('phase 2 training')

    phase_2_logdir = params['log_dir']
    phase_2_save_dir = params['save_dir']
    phase_2_training_steps = params['training_steps']
    phase_2_eval_steps = params['eval_steps']
    phase_2_save_steps = params['save_steps']
    phase_2_lr = params['lr']
    phase_2_min_lr = params['min_lr']
    phase_2_warmup_steps = params['warmup_steps']
    phase_2_anneal_steps = params['anneal_steps']
    phase_2_device = params['device']
    
    paired_model = phase_2_train(model = paired_model,
                             train_loader= train_loader,
                             test_loader= test_loader,
                             log_dir = os.path.join(phase_2_logdir, 'pair'),
                             save_dir = os.path.join(phase_2_save_dir, 'pair'),
                             training_steps= phase_2_training_steps,
                             eval_steps= phase_2_eval_steps,
                             save_steps= phase_2_save_steps,
                             lr = phase_2_lr,
                             min_lr = phase_2_min_lr,
                             warmup_steps= phase_2_warmup_steps,
                             anneal_steps= phase_2_anneal_steps,
                             device = phase_2_device)
    
    print('OVER')
    


if __name__ == "__main__":

    ## parse parameter 
    parser = argparse.ArgumentParser(description="Run SR evaluation with specified config.")
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()

    config_path = args.config_path 

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    print(config)

    eval_id = config["eval_id"]
    params = config["params"]
    
    main(eval_id, params)