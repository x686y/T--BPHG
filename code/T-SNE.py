

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear, TransformerConv, FiLMConv
import torchmetrics
import pandas as pd
import sys, os
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt
import random

from dataprogressNettcrfull import *
from config import *


class HeteroTCR(torch.nn.Module):
    def __init__(self, metadata, hidden_channels=1024, num_layers=3, net_type='SAGE', dropout_rate=0.5):
        super().__init__()
        self.encoder = HeteroGNN(metadata, hidden_channels, num_layers, net_type)
        self.decoder = MLP(hidden_channels, dropout_rate)

    def forward(self,  x_dict, edge_index_dict, edge_index_a, edge_index_b):
        z_dict = self.encoder(x_dict, edge_index_dict)
        out_tra, out_trb = self.decoder(z_dict, edge_index_a, edge_index_b)
        return out_tra, out_trb

    def get_con(self, x_dict, edge_index_dict, edge_index_a, edge_index_b):
        z_dict = self.encoder(x_dict, edge_index_dict)
        x_a = z_dict['cdr3b'][edge_index_a[0]]
        x_b = z_dict['tra_peptide'][edge_index_a[1]]
        x_tra = torch.cat([x_a, x_b], dim=1)

        x_c = z_dict['cdr3b'][edge_index_b[0]]
        x_d = z_dict['trb_peptide'][edge_index_b[1]]
        x_trb = torch.cat([x_c, x_d], dim=1)
        out_tra = x_tra 
        out_trb =x_trb
        
        out_combined = torch.cat([out_tra,out_trb], dim=1) 

        return out_combined



class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels=1024, num_layers=3, net_type='SAGE'):
        super().__init__()

        self.convs = torch.nn.ModuleList()  
        if net_type == 'SAGE': 
            for _ in range(num_layers):  
                conv = HeteroConv({
                    ('cdr3b', 'binds_to', 'tra_peptide'): SAGEConv((-1, -1), hidden_channels),
                    ('cdr3b', 'binds_to', 'trb_peptide'): SAGEConv((-1, -1), hidden_channels),
                   
                    ('tra_peptide', 'rev_binds_to', 'cdr3b'): SAGEConv((-1, -1), hidden_channels),
                    ('trb_peptide', 'rev_binds_to', 'cdr3b'): SAGEConv((-1, -1), hidden_channels),
                })
                self.convs.append(conv) 
        elif net_type == 'TF':
            for _ in range(num_layers):
                conv = HeteroConv({
                    ('cdr3b', 'binds_to', 'tra_peptide'): TransformerConv(-1, hidden_channels),
                    ('cdr3b', 'binds_to', 'trb_peptide'): TransformerConv(-1, hidden_channels),
                   
                    ('tra_peptide', 'rev_binds_to', 'cdr3b'): TransformerConv(-1, hidden_channels),
                    ('trb_peptide', 'rev_binds_to', 'cdr3b'): TransformerConv(-1, hidden_channels),
                })
                self.convs.append(conv)
        elif net_type == 'FiLM':
            for _ in range(num_layers):
                conv = HeteroConv({
                    edge_type: FiLMConv((-1, -1), hidden_channels)
                    for edge_type in metadata[1]
                })
                self.convs.append(conv)

     
    def forward(self, x_dict, edge_index_dict):
      
        initial_x_dict = x_dict.copy()
      

        # for conv in self.convs:
        for i, conv in enumerate(self.convs):  
            # print(f"--- Processing Layer {i + 1} ---")  
            for edge_type in edge_index_dict.keys():
                src, _, dst = edge_type
                # print(f"Processing edge from {src} to {dst}")
                # print(f"Checking if {src} and {dst} exist in x_dict...")
                # if src not in x_dict or dst not in x_dict:
                #     print(f"Error: {src} or {dst} not found in x_dict!")

           
            x_dict = conv(x_dict, edge_index_dict)
            # print("x_dict after conv:", x_dict.keys()) 

           
            # for key, value in x_dict.items():
            #     print(f"Node type: {key}, Features shape: {value.shape}") 
            #     print(f"Node type: {key}, Features: {value}") 

            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

         

        return x_dict



class MLP(torch.nn.Module):
    def __init__(self, hidden_channels=1024, dropout_rate=0.5):
        super().__init__()
        self.lin1 = Linear(hidden_channels * 2, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lin2 = Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)
        # Output layers modified to return additional weight per sample
        self.lin3_tra = Linear(256, 1)  # Updated: output for prediction + weight
        self.lin3_trb = Linear(256, 1)  # Updated: output for prediction + weight

        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self, z_dict, edge_index_a, edge_index_b):
        x_a = z_dict['cdr3b'][edge_index_a[0]]
        x_b = z_dict['tra_peptide'][edge_index_a[1]]
        x_tra = torch.cat([x_a, x_b], dim=1)

        x_c = z_dict['cdr3b'][edge_index_b[0]]
        x_d = z_dict['trb_peptide'][edge_index_b[1]]
        x_trb = torch.cat([x_c, x_d], dim=1)
        out_tra = x_tra 
        out_trb =x_trb
        x_tra = self.process_single_path(out_tra)
        x_trb = self.process_single_path(out_trb)

        out_tra = self.lin3_tra(x_tra)
        out_trb = self.lin3_trb(x_trb)

        # return self.sigmoid(out_tra).view(-1), self.sigmoid(out_trb).view(-1)
        return out_tra, out_trb

    def process_single_path(self, x):
        x = self.lin1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.lin2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = F.relu(x)

        return x


# config
cuda_name = 'cuda:' + args.cuda
hc = int(args.hiddenchannels)
nl = int(args.numberlayers)
nt = args.gnnnet

root_model = args.modeldir

device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
data_test, test_edge_index_a, test_edge_index_b, y_test = create_dataset_global_predict()
print(data_test)



output_dir = r"D:\xy\HeteroTCR-main\5folds_visual\picture\youGrap_Netbalfold0_tSNE"
# output_dir = r"D:\xy\HeteroTCR-main\5folds_visual\picture\youGrap_Netfullfold0_tSNE"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if __name__ == "__main__":
    test_data = pd.read_csv(test_csv, delimiter=test_delimiter)
    test_cdra, test_cdrb, test_peptide = list(test_data['tra']), list(test_data['trb']), list(test_data['peptide'])

    key_pep = []
    dict_tmp = {}
    for i in range(len(test_peptide)):
        if test_peptide[i] not in dict_tmp.keys():
            dict_tmp[test_peptide[i]] = 1
            key_pep.append(test_peptide[i])

    model = HeteroTCR(data_test.metadata(), hc, nl, nt)
    model = model.to(device)
    data_test = data_test.to(device)

    with torch.no_grad():  # Initialize lazy modules.
        out_tra, out_trb = model(data_test.x_dict, data_test.edge_index_dict, test_edge_index_a, test_edge_index_b)


    model_dir = args.testmodeldir
    val_model_path = os.path.join(root_model, model_dir)
    for root, dirs, files in os.walk(val_model_path):
        for file in files:
            if file.startswith("max"):
                PATH = os.path.join(val_model_path, file)
                model.load_state_dict(torch.load(PATH))

                model.eval()
                output_con = model.get_con(data_test.x_dict, data_test.edge_index_dict, test_edge_index_a,
                                               test_edge_index_b)
                # Detach the tensor from the computation graph and move it to CPU, then convert to NumPy
                output_con_cpu = output_con.detach().cpu().numpy()

                # Now, apply t-SNE
                tsne = TSNE(n_components=2, random_state=42)
                result = tsne.fit_transform(output_con_cpu)

                scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
                result = scaler.fit_transform(result)

                for k in range(len(key_pep)):
                    plt.figure(figsize=(12, 8))
                    plt.title('with Heterogeneous GNN module: ' + key_pep[k],fontsize=18 , fontweight='bold')
                    # plt.text(-0.05, 1.05, 'C', fontsize=22, fontweight='bold', va='center', ha='center',
                    #          transform=plt.gca().transAxes)
                    for i in range(len(y_test)):
                        if test_peptide[i] == key_pep[k]:
                            if y_test[i] == 0:
                                s1 = plt.scatter(result[i, 0], result[i, 1], c='#f57c6e', s=40)
                            elif y_test[i] == 1:
                                s2 = plt.scatter(result[i, 0], result[i, 1], c='#71b7ed', s=40)
                        else:
                            s = plt.scatter(result[i, 0], result[i, 1], c='lightgrey', s=40)
                    plt.legend((s1, s2), ('0', '1'), loc='best', title='Binder', fontsize=14, title_fontsize=16)
                    plt.xlabel("t-SNE 1", fontsize=18, fontweight='bold')
                    plt.ylabel("t-SNE 2", fontsize=18, fontweight='bold')
                    plt.tick_params(axis='both', which='major', labelsize=18) 
                    plt.savefig(os.path.join(output_dir, key_pep[k] + '.png'), format="png", dpi=300)
                    plt.close('all')
