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



