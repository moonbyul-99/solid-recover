import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import seaborn as sns
import scanpy as sc 
from scipy import sparse

def plot_heatmap(adata, catorder, gene_list, figsize = (8,8)):
    data = adata[:,gene_list].X
    if sparse.issparse(data):
        data = data.toarray()
    

    plot_x = pd.DataFrame(data = data, columns = gene_list)
    plot_x.loc[:,'cat'] = adata.obs.cell_type.values 

    # 1. 按 catorder 重排
    plot_x['cat'] = pd.Categorical(plot_x['cat'], categories=catorder, ordered=True)
    plot_x = plot_x.sort_values('cat').reset_index(drop=True)

    # 2. 提取数据和类别
    data_matrix = plot_x.iloc[:, :-1].values  # numpy array for imshow
    row_categories = plot_x['cat'].tolist()

    # 3. 创建类别颜色映射
    palette = sns.color_palette("Set2", len(catorder))  # 或 "tab10", "hls"
    cat_to_color = dict(zip(catorder, palette))

    # 4. 计算每个类别的起始和结束行索引（用于 colorbar 分块）
    cat_blocks = []
    current_cat = row_categories[0]
    start_idx = 0
    for i in range(1, len(row_categories)):
        if row_categories[i] != current_cat:
            cat_blocks.append((current_cat, start_idx, i-1))
            current_cat = row_categories[i]
            start_idx = i
    cat_blocks.append((current_cat, start_idx, len(row_categories)-1))

    # 5. 创建图形和子图（左：注释条，右：热图）
    fig, (ax_annot, ax_heatmap) = plt.subplots(1, 2, figsize=figsize, 
                                            gridspec_kw={'width_ratios': [0.03, 1]}, 
                                            sharey=True)

    # 6. 绘制左侧注释条（分块着色）
    # 创建一个 (n, 1, 3) 的 RGB 颜色数组
    color_array = np.zeros((len(row_categories), 1, 3))  # ← RGB, shape (n, 1, 3)

    for cat, start, end in cat_blocks:
        color = cat_to_color[cat]  # e.g., (0.1, 0.5, 0.9)
        color_array[start:end+1, 0, :] = color  # 广播成功！

    ax_annot.imshow(color_array, aspect='auto', origin='upper', interpolation='none',cmap = None)
    ax_annot.set_frame_on(False)
    ax_annot.set_xticks([])
    ax_annot.set_yticks([])

    # 7. 绘制主热图
    ax_heatmap.imshow(data_matrix, cmap='RdBu_r', aspect='auto', origin='upper', vmin=np.percentile(data_matrix, 1), vmax=np.percentile(data_matrix, 99))
    ax_heatmap.set_xlabel('Gene')
    ax_heatmap.set_ylabel('Cell')

    # 8. 设置 y 轴刻度和标签（只在每个区块顶部显示类别名）
    ytick_positions = []
    ytick_labels = []
    for cat, start, end in cat_blocks:
        pos = (start + end) // 2  # 区块中心位置
        ytick_positions.append(pos)
        ytick_labels.append(cat)

    ax_heatmap.set_yticks(ytick_positions, )
    ax_heatmap.set_yticklabels(ytick_labels, rotation=0, fontsize=10, fontweight='bold', )

    # 9. 添加表达量色标（右侧）
    cbar = plt.colorbar(ax_heatmap.get_images()[0], ax=ax_heatmap, shrink=0.8, pad=0.02)
    cbar.set_label('Expression')

    ax_heatmap.set_xticks(range(len(gene_list)))
    ax_heatmap.set_xticklabels(gene_list, rotation=90, ha='right', fontsize=8)

    plt.tight_layout()
    plt.grid(False)
    plt.show()