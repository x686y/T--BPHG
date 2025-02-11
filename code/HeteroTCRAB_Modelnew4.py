


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear, TransformerConv, FiLMConv, HGTConv


# 构建GNN模型

# 构建GNN模型
# 确保 HeteroTCR 中的 forward 正确调用了 MLP
class HeteroTCR(torch.nn.Module):
    def __init__(self, metadata, hidden_channels=1024, num_layers=3, net_type='SAGE', dropout_rate=0.5):
        super().__init__()
        self.encoder = HeteroGNN(metadata, hidden_channels, num_layers, net_type)
        self.decoder = MLP(hidden_channels, dropout_rate)

    def forward(self, x_dict, edge_index_dict, edge_index_a, edge_index_b):
        z_dict = self.encoder(x_dict, edge_index_dict)
        out_tra, out_trb= self.decoder(z_dict, edge_index_a, edge_index_b)
        return out_tra, out_trb

# HeteroGNN编码器通过堆叠不同类型的图卷积层处理图数据，包括不同类型的节点特征和边索引，以生成编码后的节点特征表示
class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels=1024, num_layers=3, net_type='SAGE'):
        super().__init__()

        self.convs = torch.nn.ModuleList()  # 初始化一个ModuleList，用于存储多个卷积层
        if net_type == 'SAGE':  # 根据net_type# 循环num_layers，表示消息传递的层数，也就是K阶邻域的层数，参数选择不同类型的图神经网络层进行堆叠，如果net_type为'SAGE'，则堆叠GraphSAGE风格的卷积层
            for _ in range(num_layers):  # 对每一层，根据metadata中的边类型信息创建一个HeteroConv对象，该对象内部包含针对每种边类型的SAGEConv卷积层。
                # SAGEConv((-1, -1), hidden_channels)表示输入和输出特征的维度都是动态的（-1表示自动推断），输出特征维度为hidden_channels
                # 使用 HeteroConv 创建针对每个边类型的卷积层
                conv = HeteroConv({
                    ('cdr3b', 'binds_to', 'tra_peptide'): SAGEConv((-1, -1), hidden_channels),
                    ('cdr3b', 'binds_to', 'trb_peptide'): SAGEConv((-1, -1), hidden_channels),
                    # 添加反向边，新加
                    ('tra_peptide', 'rev_binds_to', 'cdr3b'): SAGEConv((-1, -1), hidden_channels),
                    ('trb_peptide', 'rev_binds_to', 'cdr3b'): SAGEConv((-1, -1), hidden_channels),
                })
                self.convs.append(conv)  # 将创建的HeteroConv对象添加到ModuleList中
        elif net_type == 'TF':
            for _ in range(num_layers):
                conv = HeteroConv({
                    ('cdr3b', 'binds_to', 'tra_peptide'): TransformerConv(-1, hidden_channels),
                    ('cdr3b', 'binds_to', 'trb_peptide'): TransformerConv(-1, hidden_channels),
                    # 添加反向边，新加
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

        # elif net_type == 'HGT':
        #     for _ in range(num_layers):
        #         conv = HeteroConv({
        #             edge_type: HGTConv(
        #                 in_channels=hidden_channels,  # 确保这是正确的输入维度,没改出来
        #                 out_channels=hidden_channels,
        #                 heads=8, metadata=metadata
        #             ) for edge_type in metadata[1]
        #         }, aggr='mean')  # 使用合适的聚合方式
        #         self.convs.append(conv)

    # 正确实现的 forward 方法
    def forward(self, x_dict, edge_index_dict):
        # 保存初始的 x_dict 以确保节点特征不被意外删除
        initial_x_dict = x_dict.copy()
        # for i, conv in enumerate(self.convs):
        #     x_dict = conv(x_dict, edge_index_dict)
        #     if i == len(self.convs) - 1:  # 检查是否为最后一层
        #         print(f"第 {i + 1} 层之后")
        #         for node_type, features in x_dict.items():
        #             print(f"节点类型：{node_type}, 特征形状：{features.shape}")
        #     x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        #     #下面开始是运行原来的，上面是调试
        # print("x_dict keys:", x_dict.keys())  # 调试用
        # print("edge_index_dict keys:", edge_index_dict.keys())  # 调试用

        # for conv in self.convs:
        for i, conv in enumerate(self.convs):  # 在这里使用 enumerate 来获取索引,73-74新加
            # print(f"--- Processing Layer {i + 1} ---")  # 打印当前层的信息
            for edge_type in edge_index_dict.keys():
                src, _, dst = edge_type
                # print(f"Processing edge from {src} to {dst}")
                # print(f"Checking if {src} and {dst} exist in x_dict...")
                # if src not in x_dict or dst not in x_dict:
                #     print(f"Error: {src} or {dst} not found in x_dict!")

            # 应用卷积操作，并确保特征字典中的所有节点类型不会丢失
            x_dict = conv(x_dict, edge_index_dict)
            # print("x_dict after conv:", x_dict.keys())  # 确保 cdr3b 没有丢失

            # 打印每种节点类型的特征，87-89新加
            # for key, value in x_dict.items():
            #     print(f"Node type: {key}, Features shape: {value.shape}")  # 打印每种节点的特征形状
            #     print(f"Node type: {key}, Features: {value}")  # 打印每种节点的特征值

            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

            # # 如果某些特征丢失了，用初始特征恢复
            # for key in initial_x_dict.keys():
            #     if key not in x_dict:
            #         x_dict[key] = initial_x_dict[key]
            #         print(f"Restored {key} in x_dict.")

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

        out_tra = self.lin3_tra(x_tra)#输出是原始的对数值（logits），即未经缩放的分数。这些输出代表了每个样本的原始预测，
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

