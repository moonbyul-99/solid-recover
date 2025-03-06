"""
Performance evaluation metrics
"""

import torch
import scipy
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

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


def calculate_hit_rate(embeddings_a, embeddings_b, K, metric = 'euclidean'):
    """
    计算跨模态匹配准确率
    :param embeddings_a: 模态A的嵌入向量，形状 (N, d)
    :param embeddings_b: 模态B的嵌入向量，形状 (N, d)
    :param K: 近邻数
    :return: 匹配准确率
    """
    N = embeddings_a.shape[0]
    if embeddings_b.shape[0] != N:
        raise ValueError("两个模态的样本数量必须一致")
    
    count = 0
    
    # 将两个模态的embedding合并成一个大的embedding集合
    all_embeddings = np.vstack([embeddings_a, embeddings_b])  # 形状 (2N, d)
    
    # 方向1: 模态A到模态B的匹配
    nbrs_a = NearestNeighbors(n_neighbors=K + 1, metric=metric, algorithm='auto').fit(all_embeddings)  # K+1 是为了排除自身
    _, indices_a = nbrs_a.kneighbors(embeddings_a)
    for i in range(N):
        # 排除自身后检查是否命中对应的模态B的embedding
        neighbors = indices_a[i][1:]  # 排除第一个邻居（自身）
        if (i + N) in neighbors:  # 模态B的第i个embedding在模态A的K近邻中
            count += 1
    
    # 方向2: 模态B到模态A的匹配
    nbrs_b = NearestNeighbors(n_neighbors=K + 1, metric = metric, algorithm='auto').fit(all_embeddings)  # K+1 是为了排除自身
    _, indices_b = nbrs_b.kneighbors(embeddings_b)
    for i in range(N):
        # 排除自身后检查是否命中对应的模态A的embedding
        neighbors = indices_b[i][1:]  # 排除第一个邻居（自身）
        if i in neighbors:  # 模态A的第i个embedding在模态B的K近邻中
            count += 1
    
    return count / (2 * N)
