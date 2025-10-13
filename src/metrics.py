"""
Performance evaluation metrics
"""

import torch
import scipy
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
import os 
import json
import anndata as ad  
import scanpy as sc

### Matching Metrics ###
def matching_metrics(similarity=None, x=None, y=None, metric = 'euclidean', **kwargs):

    if similarity is None:
        if x.shape != y.shape:
            raise ValueError("Shapes of x and y do not match!")
        
        if metric == "cosine":
            # Compute cosine similarity
            x_norm = np.linalg.norm(x, axis=1, keepdims=True)
            y_norm = np.linalg.norm(y, axis=1, keepdims=True)
            similarity = np.dot(x, y.T) / (x_norm * y_norm.T + 1e-8)  # Add epsilon to avoid division by zero
        
        elif metric == "euclidean":
            # Compute Euclidean distance and convert to similarity
            distance = cdist(x, y, metric="euclidean", **kwargs)
            similarity = -distance  # Negative distance as similarity
        
        else:
            raise ValueError("Unsupported metric. Choose 'cosine' or 'euclidean'.")

    if not isinstance(similarity, torch.Tensor):
        similarity = torch.from_numpy(similarity)

    with torch.no_grad():
        # similarity = output.logits_per_atac
        batch_size = similarity.shape[0]
        acc_x = (
            torch.sum(
                torch.argmax(similarity, dim=1)
                == torch.arange(batch_size).to(similarity.device)
            )
            / batch_size
        )
        acc_y = (
            torch.sum(
                torch.argmax(similarity, dim=0)
                == torch.arange(batch_size).to(similarity.device)
            )
            / batch_size
        )
        foscttm_x = (
            (similarity > torch.diag(similarity)).float().mean(axis=1).mean().item()
        )
        foscttm_y = (
            (similarity > torch.diag(similarity)).float().mean(axis=0).mean().item()
        )
        # matchscore_x = similarity.softmax(dim=1).diag().mean().item()
        # matchscore_y = similarity.softmax(dim=0).diag().mean().item()
        X = similarity
        mx = torch.max(X, dim=1, keepdim=True).values
        hard_X = (mx == X).float()
        logits_row_sums = hard_X.clip(min=0).sum(dim=1)
        matchscore = hard_X.clip(min=0).diagonal().div(logits_row_sums).mean().item()

        acc = (acc_x + acc_y) / 2
        foscttm = (foscttm_x + foscttm_y) / 2
        # matchscore = (matchscore_x + matchscore_y)/2
        return acc.item(), matchscore, foscttm

def calculate_hit_rate(embeddings_a, embeddings_b, K, metric='euclidean'):
    """
    计算跨模态 top-K 匹配命中率，分别返回两个方向的结果。
    
    :param embeddings_a: 模态A的嵌入向量，形状 (N, d)，例如 RNA
    :param embeddings_b: 模态B的嵌入向量，形状 (N, d)，例如 ATAC
    :param K: 近邻数量（不包括自身）
    :param metric: 距离度量方式，如 'euclidean', 'cosine' 等
    :return: dict 包含两个方向的命中率
             {
                 'a_to_b': float,  # 模态A到模态B的命中率
                 'b_to_a': float,  # 模态B到模态A的命中率
                 'average': float  # 平均命中率
             }
    """
    N = embeddings_a.shape[0]
    if embeddings_b.shape[0] != N:
        raise ValueError("两个模态的样本数量必须一致")

    # 方向1: A → B （例如 RNA query 在 ATAC 数据库中找近邻）
    nbrs_b = NearestNeighbors(n_neighbors=K, metric=metric, algorithm='auto').fit(embeddings_b)
    indices_a2b = nbrs_b.kneighbors(embeddings_a, return_distance=False)  # 形状: (N, K)
    hits_a2b = 0
    for i in range(N):
        if i in indices_a2b[i]:  # 是否在 top-K 中找到了对应的第 i 个 B 样本
            hits_a2b += 1
    hr_a2b = hits_a2b / N

    # 方向2: B → A （例如 ATAC query 在 RNA 数据库中找近邻）
    nbrs_a = NearestNeighbors(n_neighbors=K, metric=metric, algorithm='auto').fit(embeddings_a)
    indices_b2a = nbrs_a.kneighbors(embeddings_b, return_distance=False)  # 形状: (N, K)
    hits_b2a = 0
    for i in range(N):
        if i in indices_b2a[i]:  # 是否在 top-K 中找到了对应的第 i 个 A 样本
            hits_b2a += 1
    hr_b2a = hits_b2a / N
    return (hr_a2b + hr_b2a) / 2

### Prediction Metrics ###

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

    # return {
    #     'a_to_b': hr_a2b,
    #     'b_to_a': hr_b2a,
    #     'average': (hr_a2b + hr_b2a) / 2
    # }


# def calculate_hit_rate(embeddings_a, embeddings_b, K, metric = 'euclidean'):
#     """
#     计算跨模态匹配准确率
#     :param embeddings_a: 模态A的嵌入向量，形状 (N, d)
#     :param embeddings_b: 模态B的嵌入向量，形状 (N, d)
#     :param K: 近邻数
#     :return: 匹配准确率
#     """
#     N = embeddings_a.shape[0]
#     if embeddings_b.shape[0] != N:
#         raise ValueError("两个模态的样本数量必须一致")
    
#     count = 0
    
#     # 将两个模态的embedding合并成一个大的embedding集合
#     all_embeddings = np.vstack([embeddings_a, embeddings_b])  # 形状 (2N, d)
    
#     # 方向1: 模态A到模态B的匹配
#     nbrs_a = NearestNeighbors(n_neighbors=K + 1, metric=metric, algorithm='auto').fit(all_embeddings)  # K+1 是为了排除自身
#     _, indices_a = nbrs_a.kneighbors(embeddings_a)
#     for i in range(N):
#         # 排除自身后检查是否命中对应的模态B的embedding
#         neighbors = indices_a[i][1:]  # 排除第一个邻居（自身）
#         if (i + N) in neighbors:  # 模态B的第i个embedding在模态A的K近邻中
#             count += 1
    
#     # 方向2: 模态B到模态A的匹配
#     nbrs_b = NearestNeighbors(n_neighbors=K + 1, metric = metric, algorithm='auto').fit(all_embeddings)  # K+1 是为了排除自身
#     _, indices_b = nbrs_b.kneighbors(embeddings_b)
#     for i in range(N):
#         # 排除自身后检查是否命中对应的模态A的embedding
#         neighbors = indices_b[i][1:]  # 排除第一个邻居（自身）
#         if i in neighbors:  # 模态A的第i个embedding在模态B的K近邻中
#             count += 1
    
#     return count / (2 * N)
