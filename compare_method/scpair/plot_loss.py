import re
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
def logger_extract(log_line):
    match = re.search(r'Epoch=(\d+).*?validation Loss=([\d.]+)', log_line)
    if match:
        epoch = int(match.group(1))
        val_loss = float(match.group(2))
        return [epoch, val_loss]
    else:
        return None

def plot_loss(logger_path: str, save_dir = None):
    rna_2_atac_str = "Start training feature predictor: {'input': 'Gene Expression', 'output': 'Peaks'}"
    atac_2_rna_str = "Start training feature predictor: {'input': 'Peaks', 'output': 'Gene Expression'}"
    atac_2_rna_embed_str = "Start mapping Gene Expression to Peaks embeddings"
    rna_2_atac_embed_str = "Start mapping Peaks to Gene Expression embeddings"  

    loss_1 = []
    loss_2 = []
    loss_3 = []
    loss_4 = []

    gate_1 = False
    gate_2 = False
    gate_3 = False
    gate_4 = False

    with open(logger_path, 'r') as f:
        for line in f:
            log_line = line.strip()
            if rna_2_atac_str in log_line:
                gate_1 = True 
            
            if atac_2_rna_str in log_line:
                gate_2 = True
                gate_1 = False
            
            if atac_2_rna_embed_str in log_line:
                gate_3 = True
                gate_2 = False
            
            if rna_2_atac_embed_str in log_line:
                gate_4 = True
                gate_3 = False

            info = logger_extract(log_line)
            if info is not None:
                if gate_1 and info not in loss_1:
                    loss_1.append(info)
                if gate_2 and info not in loss_2:
                    loss_2.append(info)
                if gate_3 and info not in loss_3:
                    loss_3.append(info)
                if gate_4   and info not in loss_4:
                    loss_4.append(info)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))         
    axes = axes.flatten()
    axes[0].plot(np.array(loss_1)[:,1].reshape(-1))
    axes[0].set_title('Predicting ATAC using RNA loss')
    axes[0].set_xlabel('Epoch')
    axes[1].plot(np.array(loss_2)[:,1].reshape(-1))
    axes[1].set_title('Predicting RNA using ATAC loss')
    axes[1].set_xlabel('Epoch')
    axes[2].plot(np.array(loss_3)[:,1].reshape(-1))
    axes[2].set_title('Mapping RNA to ATAC embeddings loss')
    axes[2].set_xlabel('Epoch')
    axes[3].plot(np.array(loss_4)[:,1].reshape(-1))
    axes[3].set_title('Mapping ATAC to RNA embeddings loss')
    axes[3].set_xlabel('Epoch')
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'loss.png'), bbox_inches='tight')
    plt.show()
    return None 

if __name__ == '__main__':

    for i in range(1,5):
        if i == 2:
            continue
        logger_path = f'case{i}.out'
        save_dir = f'case_{i}'
        plot_loss(logger_path, save_dir = save_dir)