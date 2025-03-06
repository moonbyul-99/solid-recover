from metrics import calculate_hit_rate, matching_metrics
from tqdm import tqdm 
import anndata as ad 
import os 
import pandas as pd 
from multiprocessing import Process

def one_sample_eval(sc_test, metric = 'cosine'):
    rna = sc_test.obsm['X_embed'][sc_test.obs.batch == 'rna']
    atac = sc_test.obsm['X_embed'][sc_test.obs.batch == 'atac']

    ### calculate top_k hit rate
    tmp = []
    for i in [1,5,10,15,20,30,50,100]:
        tmp.append(calculate_hit_rate(rna, atac, i, metric = metric))

    ### calculate metric score
    acc, ms, fs = matching_metrics(x=rna, y=atac, metric=metric)
    tmp = tmp + [acc,ms,fs]
    return tmp

def main(save_dir):
    '''
    search all .h5ad file in save dir,perform evaluation
    '''

    file_list = os.listdir(save_dir) 
    
    
    # get h5ad file only 
    h5_file = []
    for ele in file_list:
        if '.h5ad' in ele:
            h5_file.append(ele)
    #print(h5_file)
    eval_res = []
    idx = []
    
    # perform evaluation for each h5ad file 
    for ele in tqdm(h5_file):
        idx.append(ele.split('.h5ad')[0].split('_')[-1])
        path = os.path.join(save_dir, ele)
        sc_test = ad.read_h5ad(path)

        res = one_sample_eval(sc_test, metric = 'cosine')
        eval_res.append(res)
    
    columns = [f'top_{i}_hit' for i in [1,5,10,15,20,30,50,100]]
    columns = columns + ['acc', 'matchscore', 'foscttm']
    res_df =pd.DataFrame(eval_res, index = idx, columns = columns)
    res_df.to_csv(os.path.join(save_dir, 'eval_result.csv'))
    print('program over')
    return None 

def get_lowest_level_folders(directory):
    lowest_folders = []
    for root, dirs, files in os.walk(directory):
        # 如果当前文件夹没有子文件夹，则认为是最底层文件夹
        if not dirs:  # dirs 是一个列表，表示当前文件夹下的子文件夹
            lowest_folders.append(root)
    return lowest_folders


if __name__ == '__main__':
    #save_dir = ''
    #main(save_dir)
    # 示例用法
    directory_path = "/home/rsun@ZHANGroup.local/sr_project/saved_models"
    lowest_folders = get_lowest_level_folders(directory_path)
    target_folder = []
    for ele in lowest_folders:
        if 'phase_2' in ele:
            if '03-03' in ele:
                target_folder.append(ele)
    print(len(target_folder))
    print(target_folder)

    # 创建并启动多个进程
    processes = []
    for ele in target_folder:
        save_dir = ele
        process = Process(target=main, args=(save_dir,))
        processes.append(process)
        process.start()
        print(f'{ele} start')

    # 等待所有进程完成
    for process in processes:
        process.join()

    print("All processes completed.")

