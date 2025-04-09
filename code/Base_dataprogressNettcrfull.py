
import pandas as pd
import numpy as np
import os
import pickle
import torch
import torchmetrics
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

from config import *

root = os.path.join(args.pridir, args.secdir, args.terdir)
train_csv = os.path.join(root, 'train_data.tsv')
test_csv = os.path.join(root, 'test_data.tsv')
train_delimiter = '\t'
test_delimiter = '\t'

# ------------------------------------------
path_pickle_train = os.path.join(root, 'Baseglobal_train_dataset_CNNfeature.pickle')
path_pickle_test = os.path.join(root, 'Baseglobal_test_dataset_CNNfeature.pickle')
train_path_pickle_cdr3 = os.path.join(root, 'BaseCNN_feature_cdr3_train.pickle')
train_path_pickle_peptidecdr3a=os.path.join(root, 'BaseCNN_feature_peptidecdr3a_train.pickle')
train_path_pickle_peptidecdr3b=os.path.join(root, 'BaseCNN_feature_peptidecdr3b_train.pickle')
test_path_pickle_cdr3 = os.path.join(root, 'BaseCNN_feature_cdr3_test.pickle')
test_path_pickle_peptidecdr3a=os.path.join(root, 'BaseCNN_feature_peptidecdr3a_test.pickle')
test_path_pickle_peptidecdr3b=os.path.join(root, 'BaseCNN_feature_peptidecdr3b_test.pickle')
def TCRDataset_global(cdr3b_map, tra_peptide_map, trb_peptide_map, cdr3b_graph, tra_peptide_graph, trb_peptide_graph,
                      edge_index_tra, edge_index_trb):
    Graphdata = HeteroData()

    # Prepare cdr3b node features
    # cdr3b_x = torch.Tensor()
    # for cb_num in cdr3b_map.values():
    #     if cb_num not in cdr3b_graph:
    #         raise ValueError(f"Key {cb_num} not found in cdr3b_graph.")
    #     cdr3b_feature = np.array(cdr3b_graph[cb_num])
    #     cdr3b_x = torch.cat((cdr3b_x, torch.Tensor(cdr3b_feature).unsqueeze(0)), 0)
    # Graphdata['cdr3b'].x = cdr3b_x

    # # Prepare tra_peptide node features
    # tra_peptide_x = torch.Tensor()
    # for tp_num in tra_peptide_map.values():
    #     tra_peptide_feature = np.array(tra_peptide_graph[tp_num])
    #     tra_peptide_x = torch.cat((tra_peptide_x, torch.Tensor(tra_peptide_feature).unsqueeze(0)), 0)
    # Graphdata['tra_peptide'].x = tra_peptide_x
    #
    # # Prepare trb_peptide node features
    # trb_peptide_x = torch.Tensor()
    # for tb_num in trb_peptide_map.values():
    #     trb_peptide_feature = np.array(trb_peptide_graph[tb_num])
    #     trb_peptide_x = torch.cat((trb_peptide_x, torch.Tensor(trb_peptide_feature).unsqueeze(0)), 0)
    # Graphdata['trb_peptide'].x = trb_peptide_x
    #
    # # Connect cdr3b and tra_peptide nodes
    # Graphdata['cdr3b', 'binds_to', 'tra_peptide'].edge_index = edge_index_tra
    #
    # # Connect cdr3b and trb_peptide nodes
    # Graphdata['cdr3b', 'binds_to', 'trb_peptide'].edge_index = edge_index_trb
    #
    # # Ensure the graph is undirected
    # Graphdata = ToUndirected()(Graphdata)
    #
    # return Graphdata


    # Prepare cdr3b node features
    cdr3b_x = torch.Tensor()
    for cb_num in cdr3b_map.values():
        # 获取 cdr3b_map 中的实际字符串键
        cb_key = list(cdr3b_map.keys())[cb_num]

        # 检查 cb_key 是否在 cdr3b_graph 中
        if cb_key not in cdr3b_graph:
            print(f"Key {cb_key} not found in cdr3b_graph.")
            continue  # 或者可以选择 raise 错误，或者其他处理方式

        # 假设 cb_key 一定存在
        cdr3b_feature = np.array(cdr3b_graph[cb_key])
        cdr3b_x = torch.cat((cdr3b_x, torch.Tensor(cdr3b_feature).unsqueeze(0)), 0)

    Graphdata['cdr3b'].x = cdr3b_x

    # Prepare tra_peptide node features
    tra_peptide_x = torch.Tensor()
    for tp_num in tra_peptide_map.values():
    
        tp_key = list(tra_peptide_map.keys())[tp_num]

        # print("tra_peptide_graph keys:", tra_peptide_graph.keys()) 
        # print("tp_key:", tp_key) 

       
        tra_peptide_feature = np.array(tra_peptide_graph[tp_key])
        tra_peptide_x = torch.cat((tra_peptide_x, torch.Tensor(tra_peptide_feature).unsqueeze(0)), 0)

    Graphdata['tra_peptide'].x = tra_peptide_x

    # Prepare trb_peptide node features
    trb_peptide_x = torch.Tensor()
    for tb_num in trb_peptide_map.values():
       
        tb_key = list(trb_peptide_map.keys())[tb_num]

        # print("trb_peptide_graph keys:", trb_peptide_graph.keys())  
        # print("tb_key:", tb_key) 

        
        trb_peptide_feature = np.array(trb_peptide_graph[tb_key])
        trb_peptide_x = torch.cat((trb_peptide_x, torch.Tensor(trb_peptide_feature).unsqueeze(0)), 0)

    Graphdata['trb_peptide'].x = trb_peptide_x

    # Connect cdr3b and tra_peptide nodes
    Graphdata['cdr3b', 'binds_to', 'tra_peptide'].edge_index = edge_index_tra

    # Connect cdr3b and trb_peptide nodes
    Graphdata['cdr3b', 'binds_to', 'trb_peptide'].edge_index = edge_index_trb

    # Ensure the graph is undirected
    Graphdata = ToUndirected()(Graphdata)

    return Graphdata



def create_dataset_global():
    if not os.path.exists(root):
        os.makedirs(root)
   
    if os.path.exists(path_pickle_train):
        os.remove(path_pickle_train)

   
    train_data = pd.read_csv(train_csv, delimiter=train_delimiter)
 
    train_data['cdr3'] = train_data['A3'].astype(str) + train_data['B3'].astype(str)
    train_cdr3b = list(train_data['cdr3'])

    train_data['peptidecdr3a']=train_data['A3'].astype(str) + train_data['peptide'].astype(str)
    train_peptidecdr3a = list(train_data['peptidecdr3a'])

    train_data['peptidecdr3b'] = train_data['B3'].astype(str) + train_data['peptide'].astype(str)
    train_peptidecdr3b = list(train_data['peptidecdr3b'])

    train_binder = list(train_data['Binding'])

    train_cdr3b_unique_list = list(train_data['cdr3'].unique())
    train_peptidecdr3a_unique_list = list(train_data['peptidecdr3a'].unique())

    train_peptidecdr3b_unique_list = list(train_data['peptidecdr3b'].unique())

    mapping_train_cdr3 = {cdr3_name: i for i, cdr3_name in enumerate(train_cdr3b_unique_list)}
    mapping_train_peptidecdr3a = { peptider3a_name: i for i, peptider3a_name in enumerate(train_peptidecdr3a_unique_list)}
    mapping_train_peptidecdr3b = { peptider3b_name: i for i, peptider3b_name in enumerate(train_peptidecdr3b_unique_list)}

    train_src_cdr3 = [mapping_train_cdr3[train_cdr3b[cci]] for cci in range(len(train_cdr3b))]
    train_dst_peptidecdr3a = [mapping_train_peptidecdr3a[train_peptidecdr3a[ppi]] for ppi in range(len(train_peptidecdr3a))]
    train_edge_index_a = torch.tensor([train_src_cdr3, train_dst_peptidecdr3a])

    train_dst_peptidecdr3b = [mapping_train_peptidecdr3b[train_peptidecdr3b[ppi]] for ppi in range(len(train_peptidecdr3b))]
    train_edge_index_b = torch.tensor([train_src_cdr3, train_dst_peptidecdr3b])

    print("loading global train dataset...")
    if not os.path.exists(path_pickle_train):
        with open(train_path_pickle_cdr3, 'rb') as f1:
            cdr3b_graph = pickle.load(f1)

        with open(train_path_pickle_peptidecdr3a, 'rb') as f2:
            tra_peptide_map = pickle.load(f2)
        with open(train_path_pickle_peptidecdr3b, 'rb') as f3:
            trb_peptide_map = pickle.load(f3)

        train_cdr3b, train_peptidecdr3a,train_peptidecdr3b, train_binder = np.asarray(train_cdr3b), np.asarray(train_peptidecdr3a),  np.asarray(train_peptidecdr3b),np.asarray(
            train_binder)

        with open(path_pickle_train, 'wb') as f1:
            train_dataset = TCRDataset_global(cdr3b_map=mapping_train_cdr3, tra_peptide_map=mapping_train_peptidecdr3a,trb_peptide_map=mapping_train_peptidecdr3b,
                                              cdr3b_graph=cdr3b_graph, tra_peptide_graph=tra_peptide_map,trb_peptide_graph=trb_peptide_map,
                                              edge_index_tra=train_edge_index_a, edge_index_trb=train_edge_index_b)
            pickle.dump(train_dataset, f1)
        with open(path_pickle_train, 'rb') as f1:
            train_dataset = pickle.load(f1)
            print("train dataset pickle saved")
    else:
        with open(path_pickle_train, 'rb') as f1:
            train_dataset = pickle.load(f1)
        print("train dataset global pickle has loaded")
    print("train dataset global has prepared")
    
    print("x_dict keys:", train_dataset.x_dict.keys())

    # test data
    test_data = pd.read_csv(test_csv, delimiter=train_delimiter)
   
    test_data['cdr3'] = test_data['A3'].astype(str) + test_data['B3'].astype(str)
    test_cdr3b = list(test_data['cdr3'])

    test_data['peptidecdr3a']=test_data['A3'].astype(str) + test_data['peptide'].astype(str)
    test_peptidecdr3a = list(test_data['peptidecdr3a'])

    test_data['peptidecdr3b'] = test_data['B3'].astype(str) + test_data['peptide'].astype(str)
    test_peptidecdr3b = list(test_data['peptidecdr3b'])

    test_binder = list(test_data['Binding'])

    test_cdr3b_unique_list = list(test_data['cdr3'].unique())
    test_peptidecdr3a_unique_list = list(test_data['peptidecdr3a'].unique())

    test_peptidecdr3b_unique_list = list(test_data['peptidecdr3b'].unique())

    mapping_test_cdr3 = {cdr3_name: i for i, cdr3_name in enumerate(test_cdr3b_unique_list)}
    mapping_test_peptidecdr3a = { peptider3a_name: i for i, peptider3a_name in enumerate(test_peptidecdr3a_unique_list)}
    mapping_test_peptidecdr3b = { peptider3b_name: i for i, peptider3b_name in enumerate(test_peptidecdr3b_unique_list)}

    test_src_cdr3 = [mapping_test_cdr3[test_cdr3b[cci]] for cci in range(len(test_cdr3b))]
    test_dst_peptidecdr3a = [mapping_test_peptidecdr3a[test_peptidecdr3a[ppi]] for ppi in range(len(test_peptidecdr3a))]
    test_edge_index_a = torch.tensor([test_src_cdr3, test_dst_peptidecdr3a])

    test_dst_peptidecdr3b = [mapping_test_peptidecdr3b[test_peptidecdr3b[ppi]] for ppi in range(len(test_peptidecdr3b))]
    test_edge_index_b = torch.tensor([test_src_cdr3, test_dst_peptidecdr3b])


    print("loading global test dataset...")
    if not os.path.exists(path_pickle_test):
        with open(test_path_pickle_cdr3, 'rb') as f4:
            test_cdr3b_graph = pickle.load(f4)
        with open(test_path_pickle_peptidecdr3a, 'rb') as f5:
            test_tra_peptide_map = pickle.load(f5)
        with open(test_path_pickle_peptidecdr3b, 'rb') as f6:
            test_trb_peptide_map = pickle.load(f6)

        test_cdr3b, test_peptidecdr3a,test_peptidecdr3b, test_binder = np.asarray(test_cdr3b), np.asarray(test_peptidecdr3a),  np.asarray(test_peptidecdr3b),np.asarray(
            test_binder)

        with open(path_pickle_test, 'wb') as f2:
            test_dataset = TCRDataset_global(cdr3b_map=mapping_test_cdr3, tra_peptide_map=mapping_test_peptidecdr3a,trb_peptide_map=mapping_test_peptidecdr3b,
                                              cdr3b_graph=test_cdr3b_graph, tra_peptide_graph=test_tra_peptide_map,trb_peptide_graph=test_trb_peptide_map,
                                              edge_index_tra=test_edge_index_a, edge_index_trb=test_edge_index_b)
            pickle.dump(test_dataset, f2)
        with open(path_pickle_test, 'rb') as f2:
            test_dataset = pickle.load(f2)
            print("test dataset pickle saved")
    else:
        with open(path_pickle_test, 'rb') as f2:
            test_dataset = pickle.load(f2)
        print("test dataset global pickle has loaded")
    print("test dataset global has prepared")
  
    print("x_dict keys:", test_dataset.x_dict.keys())
    return train_dataset, test_dataset, train_edge_index_a,train_edge_index_b, test_edge_index_a, test_edge_index_b,train_binder, test_binder


def create_dataset_global_predict():
    if not os.path.exists(root):
        os.makedirs(root)

    # test data
    test_data = pd.read_csv(test_csv, delimiter=test_delimiter)
  
    test_data['cdr3'] = test_data['A3'].astype(str) + test_data['B3'].astype(str)
    test_cdr3b = list(test_data['cdr3'])

    test_data['peptidecdr3a']=test_data['A3'].astype(str) + test_data['peptide'].astype(str)
    test_peptidecdr3a = list(test_data['peptidecdr3a'])

    test_data['peptidecdr3b'] = test_data['B3'].astype(str) + test_data['peptide'].astype(str)
    test_peptidecdr3b = list(test_data['peptidecdr3b'])

    test_binder = list(test_data['Binding'])

    test_cdr3b_unique_list = list(test_data['cdr3'].unique())
    test_peptidecdr3a_unique_list = list(test_data['peptidecdr3a'].unique())

    test_peptidecdr3b_unique_list = list(test_data['peptidecdr3b'].unique())

    mapping_test_cdr3 = {cdr3_name: i for i, cdr3_name in enumerate(test_cdr3b_unique_list)}
    mapping_test_peptidecdr3a = { peptider3a_name: i for i, peptider3a_name in enumerate(test_peptidecdr3a_unique_list)}
    mapping_test_peptidecdr3b = { peptider3b_name: i for i, peptider3b_name in enumerate(test_peptidecdr3b_unique_list)}

    test_src_cdr3 = [mapping_test_cdr3[test_cdr3b[cci]] for cci in range(len(test_cdr3b))]
    test_dst_peptidecdr3a = [mapping_test_peptidecdr3a[test_peptidecdr3a[ppi]] for ppi in range(len(test_peptidecdr3a))]
    test_edge_index_a = torch.tensor([test_src_cdr3, test_dst_peptidecdr3a])

    test_dst_peptidecdr3b = [mapping_test_peptidecdr3b[test_peptidecdr3b[ppi]] for ppi in range(len(test_peptidecdr3b))]
    test_edge_index_b = torch.tensor([test_src_cdr3, test_dst_peptidecdr3b])


    print("loading global test dataset...")
    if not os.path.exists(path_pickle_test):
        with open(test_path_pickle_cdr3, 'rb') as f4:
            test_cdr3b_graph = pickle.load(f4)
        with open(test_path_pickle_peptidecdr3a, 'rb') as f5:
            test_tra_peptide_map = pickle.load(f5)
        with open(test_path_pickle_peptidecdr3b, 'rb') as f6:
            test_trb_peptide_map = pickle.load(f6)

        test_cdr3b, test_peptidecdr3a,test_peptidecdr3b, test_binder = np.asarray(test_cdr3b), np.asarray(test_peptidecdr3a),  np.asarray(test_peptidecdr3b),np.asarray(
            test_binder)

        with open(path_pickle_test, 'wb') as f2:
            test_dataset = TCRDataset_global(cdr3b_map=mapping_test_cdr3, tra_peptide_map=mapping_test_peptidecdr3a,trb_peptide_map=mapping_test_peptidecdr3b,
                                              cdr3b_graph=test_cdr3b_graph, tra_peptide_graph=test_tra_peptide_map,trb_peptide_graph=test_trb_peptide_map,
                                              edge_index_tra=test_edge_index_a, edge_index_trb=test_edge_index_b)
            pickle.dump(test_dataset, f2)
        with open(path_pickle_test, 'rb') as f2:
            test_dataset = pickle.load(f2)
            print("test dataset pickle saved")
    else:
        with open(path_pickle_test, 'rb') as f2:
            test_dataset = pickle.load(f2)
        print("test dataset global pickle has loaded")
    print("test dataset global has prepared")
    return test_dataset, test_edge_index_a, test_edge_index_b,test_binder
