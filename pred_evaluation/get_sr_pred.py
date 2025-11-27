import numpy as np 
import pandas as pd 
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import anndata as ad
from torch.utils.data import DataLoader 
from tqdm import tqdm
import muon as mu 
import sys 
sys.path.append('../src')
from sr_model import *
from sr_dataset import *
from load_eval_data import *

sys.path.append('../')
from pair_eval import * 
import argparse

def sr_pipe(case_id, output_dir, sr_ckpt, device = 'cuda', dataset = 'test'):
    '''load ori data'''
    # case_id = 'case_8'
    if dataset == 'train':
        logcount_path = f'/home/rsun@ZHANGroup.local/solid-recover/data/{case_id}/train.h5mu'#test.h5mu'
    else:
        logcount_path = f'/home/rsun@ZHANGroup.local/solid-recover/data/{case_id}/test.h5mu'
    mdata = mu.read_h5mu(logcount_path)

    '''get prediction result'''
    # output_dir = '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case8_wc_new_20251102_0359'
    pair_model = prepare_evaluation(output_dir)

    # sr_ckpt = 4000
    # device = 'cuda'
    batch_size = 128

    print(f'load model checkpoint {sr_ckpt}')
    ckpt_path = os.path.join(output_dir, 'models', 'ckpt_%s.pth'%sr_ckpt)
    pair_model.load_model(ckpt_path)
    model = pair_model.model

    #===prediction===
    print('🚀 Start evaluation...')
    model = pair_model.model 
    model.to(device)
    model.eval()


    ## TO DO: evaluation using mini-batch
    if dataset == 'train':
        test_dataset = pair_model.train_dataset # eval in a large dataset  #pair_model.test_dataset
    else:
        test_dataset = pair_model.test_dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    rna_pred = []
    atac_pred = []
    for batch in tqdm(test_loader):
        x1 = batch['omic_1'].to(device)
        x2 = batch['omic_2'].to(device) 
        outputs, loss_dic = model(x1,x2)

        rna2atac = outputs['x2_c_recon'].detach().cpu().numpy()
        atac2rna = outputs['x1_c_recon'].detach().cpu().numpy()

        rna_pred.append(atac2rna)
        atac_pred.append(rna2atac)
    rna_pred = np.concatenate(rna_pred, axis=0)
    atac_pred = np.concatenate(atac_pred, axis=0)
    print(f'generated rna2atac shape {rna_pred.shape}, atac2rna shape {atac_pred.shape}')

    '''save result'''
    pred_rna = ad.AnnData(rna_pred)
    pred_rna.obs = mdata['rna_count'].obs.copy()
    pred_rna.var = mdata['rna_count'].var.copy() 

    pred_atac = ad.AnnData(atac_pred)
    pred_atac.obs = mdata['peak_count'].obs.copy()
    pred_atac.var = mdata['peak_count'].var.copy()

    save_dir = f'{case_id}_sr'
    os.makedirs(save_dir, exist_ok = True)
    pred_rna.write(os.path.join(save_dir,'rna_pred_1.h5ad'))
    pred_atac.write(os.path.join(save_dir,'atac_pred_1.h5ad'))
    print('Program Over')
    return None

if __name__ == '__main__':
# 1. 初始化参数解析器
    parser = argparse.ArgumentParser(description='Run sr_pipe with custom arguments')

    # 2. 添加参数 (对应你函数需要的四个输入)
    parser.add_argument('--case_id', type=str, required=True, help='The case ID (e.g., case_8)')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to SR model output directory')
    parser.add_argument('--sr_ckpt', type=int, default=4000, help='Checkpoint number (int)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')

    # 3. 解析参数
    args = parser.parse_args()
    sr_pipe(case_id = args.case_id, 
            output_dir = args.output_dir, 
            sr_ckpt = args.sr_ckpt, 
            device = args.device)