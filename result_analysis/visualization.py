import numpy as np 
import pandas as pd
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import json 
import matplotlib.patheffects as pe # 导入路径效果，用于王冠描边
from utils import *
from match_compare_lineplot import * 


class exp_vis:

    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok = True) 

    def ablation_between_sr(self, sr_dir_dict, figure_id = None):
        '''
        Description:
        compare between multiple sr models using line plot
        
        Args: 
        sr_dir_dict: dict, key: experiment name, value: sr model output dir, eg: '/home/rsun@ZHANGroup.local/solid-recover/outputs/pair_scratch_case3/eval_result'
        figure_id: str, figure save_name
        '''
        sr_res_dict = {}
        for key in sr_dir_dict:
            dir = sr_dir_dict[key]
            sr_res = ckpt_merge(dir) 
            drop_columns = ['acc','top_15_hit', 'top_30_hit']
            sr_res = sr_res.drop(drop_columns, axis = 1)
            metrics = sr_res.columns.tolist()
            sr_res_dict[key] = sr_res

        # === 配置 ===
        n_metrics = len(metrics)
        lower_is_better = {'foscttm'}

        # === 创建子图 ===
        n_cols = min(4, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]

        # === 绘图 ===
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # 绘制两条曲线
            for key in sr_res_dict:
                res = sr_res_dict[key]
                ax.plot(res.index, res[metric], label=key, marker='o', linewidth=2)

            if metric in lower_is_better:
                title = f"{metric} ↓"
            else:
                title = f"{metric} ↑"
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(fontsize=10)

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/ablation_sr_{figure_id}.png', dpi=300, bbox_inches='tight')
        plt.show()
        return sr_res_dict

    def lineplot_compare( self, run_dir, case_id, compare_dir, figure_id = None):
        '''
        Description:
        compare sr model performance during training with other methods, 

        Args:
        run_dir: sr method run_dir  outputs/case/eval_result
        compare_dir: compare method output dir, generally is a fixed dir containing all compare methods results
        case_id: task case id case_{1}
        figure_id: figure save_name
        '''

        sr_res = ckpt_merge(run_dir)

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
        save_figure = os.path.join(self.save_dir, f'LINE_compare_{figure_id}_match_evaluation.png')
        line_plot(sr_res, compare_df, save_figure)
        print('OVER')
        self.sr_res = sr_res 
        self.compare_df = compare_df
        return sr_res, compare_df

    def barplot_compare(self, sr_ckpt, figure_id = None):
        '''
        Description:
        plot the barplot compare between different models
        Before using this function, please make sure the sr_res and compare_df are ready, i.e. run lineplot_compare first

        Args:
        sr_ckpt: the epoch of sr model we will shoe
        figure_id: figure save_name
        '''
        self.compare_df.loc['solid_recover'] = self.sr_res.loc[sr_ckpt,:]

        # -------------------------- 数据处理 --------------------------

        df = self.compare_df.copy() # 复制 DataFrame
        reverse_metric = 'foscttm' # 越小越好的指标

        # 1. 确保方法名称在列中，并命名为 'Method'
        df_indexed = df.reset_index().rename(columns={'index': 'Method'})
        df_bar_long = df_indexed.melt(
            id_vars='Method', 
            var_name='Metric',
            value_name='Value'
        )

        # 3. 专业配色方案
        CUSTOM_PALETTE = {
            'cobolt': "#DFF57D",         # 红色系
            'multivi': "#81f887",        # 蓝色系
            'scbutterfly': "#61F1D9",    # 灰色系
            'scpair': "#FDB767",         # 紫色系
            'solid_recover': "#07D0F3"   # 棕色系
        }

        # -------------------------- 绘图函数：王冠标注（已修复） --------------------------

        def annotate_best(ax, df_long, reverse_metric):
            """
            在每个指标的最佳柱子上添加王冠标记。
            修正了王冠定位逻辑，使其更健壮。
            """
            
            # 计算一个基于Y轴范围的相对偏移量
            y_min, y_max = ax.get_ylim()
            offset = (y_max - y_min) * 0.01 # 2.5% 的 Y 轴高度作为偏移

            # 获取唯一的指标名称
            for metric in df_long['Metric'].unique():
                sub_df = df_long[df_long['Metric'] == metric]
                
                # 确定最佳值 (处理越小越好/越大越好)
                if metric == reverse_metric:
                    best_value = sub_df['Value'].min()
                else:
                    best_value = sub_df['Value'].max()
                
                # 遍历所有柱子（patches）
                for p in ax.patches:
                    # 找到最佳高度的柱子
                    if np.isclose(p.get_height(), best_value, atol=1e-5):
                        
                        # 添加王冠标记
                        ax.text(p.get_x() + p.get_width() / 2.,
                                p.get_height() + offset, # 使用相对偏移量
                                '★', # 王冠符号
                                ha='center', va='bottom',
                                fontsize=14, color='gold',
                                # 添加黑色描边以保证在所有颜色背景下都可见
                                path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])


        # -------------------------- 绘制双子图（已修复） --------------------------

        # 设置子图布局：4:1 宽度比例，并调整间距
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05) 

        # --- 主图：所有指标 ---
        ax_main = fig.add_subplot(gs[0, 0])

        sns.barplot(
            x='Metric',
            y='Value',
            hue='Method',
            data=df_bar_long,
            palette=CUSTOM_PALETTE,
            errorbar=None,
            ax=ax_main
        )

        # 应用王冠标注
        annotate_best(ax_main, df_bar_long, reverse_metric)

        # 调整主图样式
        ax_main.set_title(f'Method Performance in {figure_id}', fontsize=16)
        ax_main.set_xlabel('Metric', fontsize=12)
        ax_main.set_ylabel('Value', fontsize=12)
        # 设置 Y 轴范围，为王冠预留空间
        ax_main.set_ylim(0, df_bar_long['Value'].max() * 1.05) 
        ax_main.legend().remove() # 移除主图的冗余图例

        # 设置 X 轴刻度标签的旋转和对齐
        ax_main.set_xticklabels(ax_main.get_xticklabels(), rotation=15, ha='right')
        ax_main.tick_params(axis='x', labelsize=10)


        # --- 子图：放大 foscttm ---
        ax_zoom = fig.add_subplot(gs[0, 1]) 

        # 过滤出 foscttm 数据
        df_foscttm = df_bar_long[df_bar_long['Metric'] == reverse_metric]

        sns.barplot(
            x='Metric',
            y='Value',
            hue='Method',
            data=df_foscttm,
            palette=CUSTOM_PALETTE,
            errorbar=None,
            ax=ax_zoom
        )

        # **[修复 1]** 移除子图内多余的图例
        ax_zoom.legend().remove()

        # **[修复 2]** 在子图上添加数值标签，并增加高度阈值判断
        MIN_HEIGHT_THRESHOLD = 1e-4 # 最小高度阈值，避免 0.0000 错误标签
        max_val_foscttm = df_foscttm['Value'].max()

        for p in ax_zoom.patches:
            height = p.get_height()
            
            if height > MIN_HEIGHT_THRESHOLD: # 仅对有效柱子添加标签
                # 标签位置稍微抬高
                ax_zoom.text(p.get_x() + p.get_width() / 2.,
                        height + max_val_foscttm * 0.05, 
                        f'{height:.4f}',
                        ha='center', va='bottom',
                        fontsize=9)

        # 应用王冠标注到子图
        annotate_best(ax_zoom, df_foscttm, reverse_metric)

        # 调整子图样式
        ax_zoom.set_title(f'Zoom: {reverse_metric} metric', fontsize=12)
        ax_zoom.set_xlabel(None) 
        ax_zoom.set_ylabel(None) 
        ax_zoom.tick_params(axis='y', labelleft=False) # 移除 Y 轴刻度标签

        # 设置 X 轴刻度标签的旋转和对齐
        ax_zoom.set_xticklabels(ax_zoom.get_xticklabels(), rotation=15, ha='right')
        ax_zoom.tick_params(axis='x', labelsize=10) 

        # 设置子图的Y轴范围，使其放大，并为王冠和标签预留空间
        ax_zoom.set_ylim(0, max_val_foscttm * 1.3)


        # --- 统一图例 ---
        handles, labels = ax_main.get_legend_handles_labels()
        # 将图例放在主图和子图之间的右侧空隙
        fig.legend(handles, labels, title='Method', bbox_to_anchor=(0.9, 0.85), loc='upper left')

        plt.tight_layout(rect=[0, 0, 0.9, 1]) # 调整布局以容纳图例
        plt.savefig(f'{self.save_dir}/barplot_{figure_id}.png', dpi=300, bbox_inches='tight')
        plt.show()