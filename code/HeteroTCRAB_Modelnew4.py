


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear, TransformerConv, FiLMConv, HGTConv



class HeteroTCR(torch.nn.Module):
    def __init__(self, metadata, hidden_channels=1024, num_layers=3, net_type='SAGE', dropout_rate=0.5):
        super().__init__()
        self.encoder = HeteroGNN(metadata, hidden_channels, num_layers, net_type)
        self.decoder = MLP(hidden_channels, dropout_rate)

    def forward(self, x_dict, edge_index_dict, edge_index_a, edge_index_b):
        z_dict = self.encoder(x_dict, edge_index_dict)
        out_tra, out_trb= self.decoder(z_dict, edge_index_a, edge_index_b)
        return out_tra, out_trb


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

        x_tra = self.process_single_path(x_tra)
        x_trb = self.process_single_path(x_trb)

        out_tra = self.lin3_tra(x_tra)
        out_trb = self.lin3_trb(x_trb)

       
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

