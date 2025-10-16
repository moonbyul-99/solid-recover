import numpy as np
import muon as mu 
import scanpy as sc
import os 
from cobolt.utils import SingleData, MultiomicDataset
def load_multi_dt(train_data_path, test_data_path):

    ###
    #   load data,
    ###

    train_mdata = mu.read_h5mu(train_data_path)
    test_mdata = mu.read_h5mu(test_data_path)

    train_atac = train_mdata.mod['peak_count']
    train_rna = train_mdata.mod['rna_count']

    test_atac = test_mdata.mod['peak_count']
    test_rna = test_mdata.mod['rna_count']

    ###
    #  Create cobolt dataset
    ###


    rna_test = SingleData(feature_name = 'GeneExpr', 
                          dataset_name = 'test_rna', 
                          feature = test_rna.var.index.values, 
                          count = test_rna.X, 
                          barcode = test_rna.obs.index.values)
    atac_test = SingleData(feature_name = 'ChromAccess', 
                           dataset_name = 'test_atac', 
                           feature = test_atac.var.index.values, 
                           count = test_atac.X, 
                           barcode = test_atac.obs.index.values)


    rna_train = SingleData(feature_name = 'GeneExpr', 
                           dataset_name = 'train', 
                           feature = train_rna.var.index.values, 
                           count = train_rna.X, 
                           barcode = train_rna.obs.index.values)
    atac_train= SingleData(feature_name = 'ChromAccess', 
                           dataset_name = 'train', 
                           feature = train_atac.var.index.values, 
                           count = train_atac.X, 
                           barcode = train_atac.obs.index.values)

    ###
    #  generate multiomic dataset
    ###

    multi_dt = MultiomicDataset.from_singledata(
        rna_test, atac_test, rna_train, atac_train)
    print(multi_dt)
    return multi_dt 