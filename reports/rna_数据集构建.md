RNA 数据集构建

1. 获取scRNA数据集
   获取全部小鼠数据集链接，写入 h5ad.txt文件中，使用wget 下载数据
   
```python
    import os
    import requests
    from bs4 import BeautifulSoup

    # 目标URL
    url = "https://data.nemoarchive.org/other/grant/aibs_internal/zeng/transcriptome/scell/10x_v3/mouse/processed/counts/"

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
        if link.endswith('.h5ad.tar'):
            links.append(url + link)

    # 打印找到的链接
    print(f"Found {len(links)} .h5ad files:")
    for link in links:
        print(link)

    # 将链接保存到文件中（可选）
    with open('h5ad_links.txt', 'w') as f:
        for link in links:
            if 'raw' in link and 'AIT17.0.rawcount' not in link :
                f.write(f"{link}\n")

    print("Links have been saved to h5ad_links.txt")
```

2. 重做rna_seq数据(make_rna_dataset.py)

    - 获取下载的.h5ad数据路径
    - 读取注释数据，保留需要的注释字段，修改注释数据index与.h5ad数据obs.index一致。同时为需要进行分类的标签训练label_encoder并保存。
    - 读取var数据。这里是为了统一全部数据的列名一致，因此保存一个.h5ad数据的var信息做为基准，后续全部数据与这个var信息对齐。（var信息保存参考1_rna_data.ipynb）
    - 依次处理.h5ad数据，选取公共obs.index,将注释信息加入到.h5ad数据中，之后将.h5ad数据逐条生成单细胞数据，通过datasets库保存数据 

3. 处理为预训练所需数据格式(rna_train_dataset.py)
   将前面的数据concat成一个统一的数据，通过collate_fn去除冗余列，将标签列通过前面的label_encoder进行编码，将feature进行1e4归一化和lop1p变换，最后写入数据中