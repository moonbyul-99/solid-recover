import numpy as np 
import pandas as pd
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import json 



def ckpt_merge(run_dir):

    res = {}
    ckpt_lists = os.listdir(run_dir)
    for ckpt in ckpt_lists:
        ckpt_point = int(ckpt.split('_')[0])
        metric_path = os.path.join(run_dir, ckpt, 'match_metric.json')
        with open(metric_path, 'r') as f:
            metric_dic = json.load(f)
        res[ckpt_point] = metric_dic
    df = pd.DataFrame.from_dict(res)
    df = df.T 
    df.sort_index(inplace=True)
    return df 

'''AUPRC calculatin'''
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
from scipy import sparse

def compute_metric_chunk(col_indices, X_true, X_pred, metric='auprc'):
    """
    对给定的列索引列表，逐列计算 average precision score。
    
    参数:
        col_indices (list or array): 要计算的列索引，如 [0, 1, 2, ..., 99]
        X_true (scipy.sparse matrix): 真实标签矩阵，shape=(n_cells, n_genes)
        X_pred (scipy.sparse matrix): 预测分数矩阵，shape=(n_cells, n_genes)
    
    返回:
        list: 按 col_indices 顺序的 AP 分数列表
    """
    results = []
    for i in col_indices:
        y_pred = X_pred[:, i].reshape(-1)
        y_true = (X_true[:, i].reshape(-1) > 0).astype(int)
        if metric == 'auroc':
            ap = roc_auc_score(y_true, y_pred)
        elif metric == 'auprc':
            ap = average_precision_score(y_true, y_pred)
        else:
            raise ValueError(f"Invalid metric: {metric}")
        results.append(ap)
    return results

from joblib import Parallel, delayed
from tqdm import tqdm
import math

def compute_metric_chunked(method_name, method_dic, n_jobs=32, metric = 'auprc'):
    """
    将列索引按 n_jobs 切分成 chunks，每个 chunk 在一个进程中计算，
    避免传递 AnnData 对象，只传递其 .X（稀疏矩阵）。
    
    参数:
        method_name (str): 如 'mvi'
        method_dic (dict): {name: [raw_adata, pred_adata]}
        n_jobs (int): 并行进程数
        total_genes (int): 要计算的基因数（列数）
    
    返回:
        list: 长度为 total_genes 的 AP 分数，顺序正确
    """
    X_true, X_pred= method_dic[method_name]
    

    p = X_true.shape[1]
    all_cols = list(range(p))
    

    chunk_size = math.ceil(len(all_cols) / n_jobs)
    chunks = [all_cols[i:i + chunk_size] for i in range(0, len(all_cols), chunk_size)]
    

    results_list = Parallel(n_jobs=n_jobs)(
        delayed(compute_metric_chunk)(chunk, X_true, X_pred, metric)
        for chunk in tqdm(chunks, desc=f"Processing {method_name} in {len(chunks)} chunks")
    )
    

    final_result = [ap for chunk_res in results_list for ap in chunk_res]
    
    return final_result