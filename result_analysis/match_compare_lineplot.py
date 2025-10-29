import numpy as np 
import pandas as pd
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import json 
from utils import *

def line_plot(sr_res, compare_df, save_figure):
    '''
    sr_res: sr method result, index: different epoch chpt. columns: different metrics
    compare_df: compare method result, index: different method. columns: different metrics
    '''

    # === 用户配置 ===
    lower_is_better_metric = sr_res.columns[-1]

    # === 显式颜色映射（请根据你的方法名填写）===
    color_map = {
        'solid-recover': '#1f77b4',   # 蓝色（主方法）
        'cobolt':       '#ff7f0e',   # 橙色
        'multivi':       '#2ca02c',   # 绿色
        'scbutterfly':       '#d62728',   # 红色
        'scpair':       '#9467bd',
    }   # 紫色

    methods = compare_df.index.tolist()
    all_methods = ['solid-recover'] + methods


    # === 准备数据 ===
    metrics = sr_res.columns.tolist()
    n_metrics = len(metrics)

    if n_metrics > 8:
        raise ValueError("Current layout supports up to 8 metrics (2x4 grid).")

    # === 创建 2x4 子图 ===
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=False)
    axes = axes.flatten()

    legend_handles = []
    for i, metric in enumerate(metrics):
        ax = axes[i]

        # 确定箭头方向
        if metric == lower_is_better_metric:
            title = f"{metric} ↓"
        else:
            title = f"{metric} ↑"

        # 1. 绘制我们的方法（实线）
        line_ours = ax.plot(
            sr_res.index,
            sr_res[metric],
            color=color_map['solid-recover'],
            marker='o',
            linewidth=2,
            markersize=4,
            label='solid-recover'
        )
        if i == 0:
            legend_handles.append(line_ours[0])

        # 2. 绘制对比方法（虚线）
        for method in methods:
            hline = ax.axhline(
                y=compare_df.loc[method, metric],
                color=color_map[method],
                linestyle='--',
                linewidth=2,
                alpha=1,
                label=method
            )
            if i == 0:
                legend_handles.append(hline)

        # 样式设置
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Steps', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.tick_params(labelsize=9)

    # === 创建统一图例（放在右侧） ===
    fig.subplots_adjust(right=0.9)  # 为图例留出足够空间
    legend_ax = fig.add_axes([0.9, 0.1, 0.12, 0.8])  # [left, bottom, width, height]
    legend_ax.axis('off')

    # 添加图例
    legend_ax.legend(
        legend_handles,
        all_methods,
        loc='center left',
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=11,
        title='Methods',
        title_fontsize=12
    )

    # === 布局调整与保存 ===
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # 留出右侧空间
    plt.savefig(save_figure, dpi=300, bbox_inches='tight')
    plt.show()
def line_plot_compare( run_dir, compare_dir, case_id, figure_id = None):
    '''
    run_dir: sr method run_dir  outputs/case/eval_result
    compare_dir: compare method output dir
    case_id: task case id
    '''

    # output_dir = '../outputs'
    #run_dir = os.path.join(run_dir, 'pair_scratch_case1','eval_result')
    sr_res = ckpt_merge(run_dir)
    #sr_res.drop(['acc'], axis = 1, inplace = True) 
    if figure_id is None:
        figure_id = case_id
    # compare_dir = '../compare_method'
    # case_id = 'case_1'
    '''
    cobolt_res
    '''
    with open(os.path.join(compare_dir, 'cobolt',case_id, 'match_metric.json'), 'r') as f:
        cob_res = json.load(f)

    '''
    MVI_res
    '''
    with open(os.path.join(compare_dir, 'multivi',case_id, 'match_metric.json'), 'r') as f:
        mvi_res = json.load(f)

    '''
    scb_res
    '''
    with open(os.path.join(compare_dir, 'scb',case_id, 'match_metric.json'), 'r') as f:
        scb_res = json.load(f) 

    '''
    scp_res 
    ''' 
    with open(os.path.join(compare_dir, 'scpair',case_id, 'match_metric.json'), 'r') as f:
        scp_res = json.load(f) 


    drop_columns = ['acc', 'top_15_hit', 'top_30_hit']
    sr_res = sr_res.drop(drop_columns, axis = 1)

    compare_df = pd.DataFrame([cob_res, mvi_res, scb_res, scp_res], index = ['cobolt', 'multivi', 'scbutterfly', 'scpair'])
    compare_df = compare_df.drop(drop_columns, axis = 1)

    '''
    modify matchscore
    ''' 

    compare_df = compare_df.rename(columns={'mathscore': 'matchscore'})
    sr_res = sr_res.rename(columns={'mathscore': 'matchscore'})

    '''
    plot figure
    '''
    os.makedirs('match_evaluation', exist_ok=True)
    save_figure = os.path.join('match_evaluation', f'LINE_case{figure_id}_match_evaluation.png')
    line_plot(sr_res, compare_df, save_figure)
    print('OVER')
    return sr_res, compare_df

if __name__ == '__main__':
    # output_dir = '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case1/eval_result'
    # compare_dir = '/home/rsun@ZHANGroup.local/solid-recover/compare_method'
    # case_id = 'case_1'

    # output_dir = '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case3/eval_result'
    # compare_dir = '/home/rsun@ZHANGroup.local/solid-recover/compare_method'
    # case_id = 'case_3'

    # output_dir = '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case4_20251012_1650/eval_result'
    # compare_dir = '/home/rsun@ZHANGroup.local/solid-recover/compare_method'
    # case_id = 'case_4'
    # figure_id = 'case4'

    # output_dir = '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case4_20251013_0338/eval_result'
    # compare_dir = '/home/rsun@ZHANGroup.local/solid-recover/compare_method'
    # case_id = 'case_4'
    # figure_id = 'case4_20251013_0338'

    # output_dir = '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case4_20251013_0747/eval_result'
    # compare_dir = '/home/rsun@ZHANGroup.local/solid-recover/compare_method'
    # case_id = 'case_4'
    # figure_id = 'case4_20251013_0747'

    # output_dir = '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case4_20251013_0754/eval_result'
    # compare_dir = '/home/rsun@ZHANGroup.local/solid-recover/compare_method'
    # case_id = 'case_4'
    # figure_id = 'case4_20251013_0754'

    # output_dir = '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case4_20251013_1009/eval_result'
    # compare_dir = '/home/rsun@ZHANGroup.local/solid-recover/compare_method'
    # case_id = 'case_4'
    # figure_id = 'case4_20251013_1009'

    # output_dir = '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case4_20251013_1011/eval_result'
    # compare_dir = '/home/rsun@ZHANGroup.local/solid-recover/compare_method'
    # case_id = 'case_4'
    # figure_id = 'case4_20251013_1011'

    # output_dir = '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case4_20251013_1124/eval_result'
    # compare_dir = '/home/rsun@ZHANGroup.local/solid-recover/compare_method'
    # case_id = 'case_4'
    # figure_id = 'case4_20251013_1124'

    # output_dir = '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case4_20251014_0134/eval_result'
    # compare_dir = '/home/rsun@ZHANGroup.local/solid-recover/compare_method'
    # case_id = 'case_4'
    # figure_id = 'case4_20251014_0134'

    output_dir = '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case4_20251014_0205/eval_result'
    compare_dir = '/home/rsun@ZHANGroup.local/solid-recover/compare_method'
    case_id = 'case_4'
    figure_id = 'case4_20251014_0205'

    # output_dir = '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case3_20251013_0339/eval_result'
    # compare_dir = '/home/rsun@ZHANGroup.local/solid-recover/compare_method'
    # case_id = 'case_3'
    # figure_id = 'case3_trainable_t'

    # output_dir = '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case2/eval_result'
    # compare_dir = '/home/rsun@ZHANGroup.local/solid-recover/compare_method'
    # case_id = 'case_2'
    # figure_id = 'case2'
    line_plot_compare( output_dir , compare_dir, case_id,figure_id = figure_id)