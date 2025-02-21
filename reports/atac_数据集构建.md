atac 数据集构建 

1. 获取原数据 

获取数据下载链接,之后wget下载全部数据至 h5ad_data目录下

```python
    import os
    import requests
    from bs4 import BeautifulSoup

    # 目标URL
    url = "http://catlas.org/renlab_downloads/wholemousebrain/snapatac2_anndata_per_sample_with_fragment_dir/"

    # 发送HTTP请求获取网页内容
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to retrieve the page.")
        exit()

    # 解析HTML内容
    soup = BeautifulSoup(response.text, 'html.parser')

    # 查找所有的 .h5ad 文件链接
    links = []
    for a_tag in soup.find_all('a', href=True):
        link = a_tag['href']
        if link.endswith('.h5ad'):
            links.append(url + link)

    # 打印找到的链接
    print(f"Found {len(links)} .h5ad files:")
    for link in links:
        print(link)

    # 将链接保存到文件中（可选）
    with open('h5ad_links.txt', 'w') as f:
        for link in links:
            f.write(f"{link}\n")

    print("Links have been saved to h5ad_links.txt")
```

2. 查看数据基本情形，确定数据处理逻辑 (1_data_check_1.ipynb)
3. 将atac数据处理成gene activity 数据 (get_ga.py) 

    使用snapatac2 中的snap.read函数读取atac数据，注意这个函数默认参数为'r+'，即对读取数据的操作都会写入原数据中，设置为'r'为只读，可以规避这点。之后将.obsm中的 'insertion' 改成'fragment_paired'，这一步主要是snapatac2版本的问题，2.7版本不接受'insertion'键，但是通过'insertion'下的矩阵数值可以确定为'fragment_paired'情形下的测序数据。之后调用snap.pp.make_gene_matrix函数获取gene activity数据。最后将gene activity 数据写入save_dir下。

    ```python
    if __name__ == '__main__':

        h5ad_dir = '../h5ad_data'
        save_dir = '../ga_data'
        os.makedirs(save_dir, exist_ok=True)

        res = os.listdir('../h5ad_data')

        ga_res = os.listdir('../ga_data')

        for ele in tqdm(res): 
            if ele in ga_res:
                continue
            
            try:
                file_path = os.path.join(h5ad_dir, ele)

                scdata = snap.read(file_path)
                scdata.obsm['fragment_paired'] = scdata.obsm['insertion']
                ga = snap.pp.make_gene_matrix(scdata, gene_anno = snap.genome.mm10) 
                #scdata.file.close()

                save_path = os.path.join(save_dir, ele) 
                ga.write(save_path)
            except:
                print(f'Error in {ele}')
                continue
                
            

        print('OVER')
    ```
4. 检查atac数据的注释信息 (2_check_anno.ipynb) 

    检查atac数据的注释信息，主要关注不同的细胞注释信息，为每个标签创建一个label encoder并保存。
5. 将atac数据的注释信息写入gene activity数据中(ga_obs.py) 

    add_obs 执行如下操作：
    - 读取gene activity数据
    - 根据一个预先定义的gene_list选择给定特征。这是因为snapatac2转出的Gene activity数据中有大约5万个基因名称，而通常的10X数据在mm10注释后的基因数目应该是32285。这个gene_list是预先定义的，例如可以取snapatac2转出的基因和10X小鼠基因的交集。
    - 根据gene activity数据的id，从注释df中筛选该数据集对应的sub_df 
    - 对sub_df, 用细胞barcode作为行索引，和gene activity数据对应，同时仅保留关心的注释标签
    - 将对应的注释信息写入gene activity中的obs
    - 将修改后的gene activity替换原有的gene activity数据
```python
def add_obs(ga_path,
            sample_name,
            gene_list, 
            anno_df):
    '''
    Args:
        ga_path: path of ga_file 
        sample_name: str, the key to select anno in annotation
        gene_list: gene_list, some feature in ga are not common gene
        anno_df: anno_df
    '''


    mutual_gene = gene_list
    df = anno_df

    scdata = sc.read_h5ad(ga_path)
    scdata.var.loc[:,'gene_symbols'] = scdata.var.index.values 

    scdata = scdata[:,mutual_gene]


    sub_df = df.iloc[(df.loc[:,'Sample'] == sample_name).values,:]
    sub_df.index = sub_df.loc[:,'Barcode'].values 
    sub_df = sub_df.loc[:,['L1','L2','L3','L4','NeuronTransmitter','Subclass']]

    print(f'anno size {sub_df.shape[0]}, ga size {scdata.shape[0]}')

    mutual_idx = np.intersect1d(scdata.obs.index.values, sub_df.index.values)
    sub_df = sub_df.loc[mutual_idx, :]
    scdata= scdata[mutual_idx, :]
    print(f' ga size {scdata.shape[0]}')

    for key in sub_df.columns:
        scdata.obs.loc[:,key] = sub_df.loc[:,key].values

    scdata.write(ga_path)

    return None

if __name__ == '__main__': 

    # add gene list, gene_path: path of gene path, some feature in ga are not common gene
    gene_path = '../10x_data/mutual_gene.npy'
    mutual_gene = np.load(gene_path, allow_pickle=True) 

    # add annotation file, anno_path: path of annotation file
    anno_path = '/home/rsun@ZHANGroup.local/atac_pretrain/anno_data/anno.csv'
    df = pd.read_csv(anno_path)

    ga_dir= '../ga_data'
    res = os.listdir(os.path.join(ga_dir)) 
    error_list = []

    for key in tqdm(res):
        if key == 'CEMBA200910_8H_rm_dlt.h5ad':
            continue
        print(key + '===='*20)
        ga_path = os.path.join(ga_dir, key)
        sample_name = key.split('_rm_dlt')[0]

        try:
            add_obs(ga_path = ga_path,
                    sample_name = sample_name,
                    gene_list= mutual_gene,
                    anno_df= df)
        except:
            error_list.append(key)
        
        #break
    print(f'OVER with {len(error_list)} error file')
    print(error_list)

```

6. 借助datasets库构建数据 (ga_datasets.py)

    对于上百万个单细胞数据，将其存入一个.h5ad文件，之后全部读取构建dataloader是很困难的。datasets库提供了高效的数据集构建策略，因此，借助datasets库来构建数据集。在构造的数据集中，每一条数据代表一个细胞，在'feature'字段保存基因表达向量，在'barcode'字段保存cell id。这两个字段是最重要的，一个记录数据，一个记录细胞原始的索引。其他字段记录其他meta info，例如细胞注释。
    
    最后，借助 Dataset.from_generator() 构造数据集并保存到内存中。

```python
def gen_datalist(scdata):
    counts = scdata.X.toarray().astype(np.int16)
    obs = scdata.obs.copy()
    res = []

    for i in range(scdata.shape[0]):
        data = {}
        data['feature'] = counts[i,:]

        meta_info = dict(obs.iloc[i,:])
        data = data | meta_info 

        data['barcode'] = scdata.obs.index.values[i]
        
        res.append(data)
    return res

def gen(datalist):
    for line in datalist:
        yield line 

if __name__ == '__main__':

    ga_dir = '../ga_data'
    res = os.listdir(ga_dir)
    save_dir = '../hf_data/atac_data'

    for file in tqdm(res):
        if file == 'CEMBA200910_8H_rm_dlt.h5ad':
            continue
        ga_path = os.path.join(ga_dir, file)

        sample_name = file.split('_rm_dlt')[0]
        scdata = sc.read_h5ad(ga_path)
        scdata.obs.loc[:,'sample'] = sample_name
        datalist = gen_datalist(scdata)
        ds = Dataset.from_generator(gen, gen_kwargs = {'datalist' : datalist})
        
        save_path = os.path.join(save_dir, sample_name)


        ds.save_to_disk(dataset_path = save_path)

    print('OVER')
```

7. concat数据，删除对训练无效的字段(ga_dataloader.py)

   前面的步骤中，我们为每一个gene activity数据生成了数据，现在，我们把所有的数据合成一个统一的训练数据集，并且只保留必要的训练字段。同时，1e4标准化和log1p的操作也在这一步实现。这样后面训练时，不需要在dataloader中进行额外的collate_fn操作。
