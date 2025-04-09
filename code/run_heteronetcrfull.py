

# import torch
# import torch.nn.functional as F
# import torchmetrics
# import pandas as pd
# import sys, os
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# # from HeteroTCRAB_Model import *
# from HeteroTCRAB_Model import *
# from data_processtcrAB import *
# from config import *
#
# # config
# NUM_EPOCHS = int(args.epochs)
# cuda_name = 'cuda:' + args.cuda
# lr = float(args.gnnlearningrate)
# wd = float(args.weightdecay)
# hc = int(args.hiddenchannels)
# nl = int(args.numberlayers)
# nt = args.gnnnet
# dropout = float(args.dropout)
# root_model = args.modeldir
# model_dir = str(args.secdir) + '_' + str(args.terdir) + '_HeteroAB'
# model_dir2 = str(args.secdir) + '_' + str(args.terdir)
# save_model_path = os.path.join(root_model, model_dir)
# root_history = args.hisdir
# device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')

import torch
import torch.nn.functional as F
import torchmetrics
import pandas as pd
import sys, os
from torchmetrics.classification import AveragePrecision
from sklearn.utils.class_weight import compute_class_weight
from HeteroTCRAB_Modelnew4 import *
from dataprogressNettcrfull import *
# from Base_dataprogressNettcrfull import *
from config import *

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# config
NUM_EPOCHS = int(args.epochs)
cuda_name = 'cuda:' + args.cuda
lr = float(args.gnnlearningrate)
wd = float(args.weightdecay)
hc = int(args.hiddenchannels)
nl = int(args.numberlayers)
nt = args.gnnnet
dropout=float(args.dropout_rate)
root_model = args.modeldir
# model_dir = str(args.secdir) + '_' + str(args.terdir) + '_Base_HeteroAB'
model_dir = str(args.secdir) + '_' + str(args.terdir) + '_GuoAndXiacaiyang_HeteroAB'


model_dir2 = str(args.secdir) + '_' + str(args.terdir)
save_model_path = os.path.join(root_model, model_dir)

root_history = args.hisdir

device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
data_train, data_test,train_edge_index_a,train_edge_index_b, test_edge_index_a, test_edge_index_b, y_train, y_test = create_dataset_global()
# print(f"y_train shape: {y_train.shape}")
# print(data_train)
# print(data_test)
# print(train_edge_index.size())

#print(len(train_cdr3b)) 
# print('train_edge_index_a',train_edge_index_a.shape)
# print('12345')
# Initialize metrics using class interface with multiclass set to True

def specificity(y_pred, y_true):
    y_pred = y_pred.round() 
    tn = ((y_pred == 0) & (y_true == 0)).sum().float()  
    fp = ((y_pred == 1) & (y_true == 0)).sum().float() 
    return tn / (tn + fp + 1e-8) 
train_f1 = torchmetrics.F1(num_classes=2, average='macro', multiclass=True).to(device)
train_precision_metric = torchmetrics.Precision(num_classes=2, average='macro', multiclass=True).to(device)
train_recall_metric = torchmetrics.Recall(num_classes=2, average='macro', multiclass=True).to(device)

test_f1 = torchmetrics.F1(num_classes=2, average='macro', multiclass=True).to(device)
test_precision_metric = torchmetrics.Precision(num_classes=2, average='macro', multiclass=True).to(device)
test_recall_metric = torchmetrics.Recall(num_classes=2, average='macro', multiclass=True).to(device)

def train(model):
  
    # print('Training on {} samples for tra_peptide...'.format(len(data_train['cdr3b', 'binds_to', 'tra_peptide'].edge_index[0])))
    # print('Training on {} samples for trb_peptide...'.format(len(data_train['cdr3b', 'binds_to', 'trb_peptide'].edge_index[0])))


    loss_fn = torch.nn.BCELoss()
  

    model.train()
    optimizer.zero_grad()
    outa, outb  = model(data_train.x_dict, data_train.edge_index_dict, train_edge_index_a, train_edge_index_b)

    out_tra_b = torch.stack([outa, outb], dim=-1)
    # print(f"out_tra_b shape: {out_tra_b.shape}")
    softmax_out_tra_b = torch.softmax(out_tra_b, dim=-1)
    # print(f"softmax_out_tra_b shape: {softmax_out_tra_b.shape}")
    # Extract the weights
    weight_tra = softmax_out_tra_b[..., 0]  
   
    # print(f"weight_tra shape: {weight_tra.shape}")
    weight_trb = softmax_out_tra_b[..., 1]  
    out=(outa+outb)/2
  

    out = out.squeeze(-1)
    # print(f"outshape: {out.shape}")



   
    out = torch.sigmoid(out)
    train_loss = loss_fn(out, torch.tensor(y_train).float().to(device))
  

    train_loss.backward()
    optimizer.step()

    train_binary_accuracy = torchmetrics.functional.accuracy(out, torch.tensor(y_train).int().to(device))
    train_ROCAUC = torchmetrics.functional.auroc(out, torch.tensor(y_train).int().to(device))
    # Update metrics using class interface
    train_precision = train_precision_metric(out, torch.tensor(y_train).int().to(device))
    train_recall = train_recall_metric(out, torch.tensor(y_train).int().to(device))
    train_f1_score = train_f1(out, torch.tensor(y_train).int().to(device))
    train_specificity = specificity(out, torch.tensor(y_train).int().to(device))

    model.eval()
    with torch.no_grad():
        outa_test, outb_test= model(data_test.x_dict, data_test.edge_index_dict, test_edge_index_a, test_edge_index_b)

        out_test_tra_b = torch.stack([outa_test, outb_test], dim=-1)
        softmax_out_test_tra_b  = torch.softmax(out_test_tra_b, dim=-1)  

       
        test_weight_tra = softmax_out_test_tra_b[..., 0]
        test_weight_trb = softmax_out_test_tra_b[..., 1]

        out_test = (outa_test + outb_test) / 2 
       
        out_test =  out_test .squeeze(-1)
        
        out_test = torch.sigmoid(out_test)
        test_loss = loss_fn(out_test, torch.tensor(y_test).float().to(device))
        # test_loss = weighted_binary_crossentropy_test(torch.tensor(y_test).to(device), out_test)

        test_binary_accuracy = torchmetrics.functional.accuracy(out_test, torch.tensor(y_test).int().to(device))
        test_ROCAUC = torchmetrics.functional.auroc(out_test, torch.tensor(y_test).int().to(device))

        # Update metrics using class interface
        test_precision = test_precision_metric(out_test, torch.tensor(y_test).int().to(device))
        test_recall = test_recall_metric(out_test, torch.tensor(y_test).int().to(device))
        test_f1_score = test_f1(out_test, torch.tensor(y_test).int().to(device))
        test_specificity = specificity(out_test, torch.tensor(y_test).int().to(device))


    return (weight_tra, weight_trb, train_loss, train_binary_accuracy, train_ROCAUC, train_precision, train_recall,
            train_f1_score, train_specificity,
            test_weight_tra, test_weight_trb, test_loss, test_binary_accuracy, test_ROCAUC, test_precision, test_recall,
            test_f1_score, test_specificity)

if __name__ == "__main__":
    model = HeteroTCR(data_train.metadata(), hc, nl, nt, dropout)
    model = model.to(device)
    data_train = data_train.to(device)
    data_test = data_test.to(device)

    with torch.no_grad():  # Initialize lazy modules.
        outa, outb= model(data_train.x_dict, data_train.edge_index_dict, train_edge_index_a, train_edge_index_b)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
  
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    epoch = []
    loss = []
    acc = []
    auc_roc = []
    precision = []
    recall = []
    f1_score = []
    specificity_list = []
    val_loss = []
    val_acc = []
    val_auc_roc = []
    val_precision_list = []  # Initialize as a list
    val_recall_list = []  # Initialize as a list
    val_f1_score_list = []  # Initialize as a list
    val_specificity_list = []

    for ep in range(1, NUM_EPOCHS + 1):
        (weight_tra, weight_trb, train_loss, train_binary_accuracy, train_ROCAUC, train_precision, train_recall,
         train_f1_score,
         train_specificity, test_weight_tra, test_weight_trb, valid_loss, val_binary_accuracy, val_ROCAUC,
         val_precision_tensor, val_recall_tensor, val_f1_score_tensor, val_specificity) = train(model)

        print(
            'Train epoch: {} - loss: {:.4f} - binary_accuracy: {:.4f} - ROCAUC: {:.4f} - precision: {:.4f} - recall: {:.4f} - f1_score: {:.4f} - specificity: {:.4f}  - val_loss: {:.4f} - val_binary_accuracy: {:.4f} - val_ROCAUC: {:.4f}  - val_precision: {:.4f} - val_recall: {:.4f} - val_f1_score: {:.4f} - val_specificity: {:.4f}'.format(
                ep, train_loss.item(), train_binary_accuracy, train_ROCAUC,train_precision.item(),
                train_recall.item(), train_f1_score.item(), train_specificity.item(), valid_loss.item(),
                val_binary_accuracy.item(), val_ROCAUC.item(), val_precision_tensor.item(),
                val_recall_tensor.item(), val_f1_score_tensor.item(), val_specificity.item() )
        )

        if not os.path.exists(root_model):
            os.makedirs(root_model)
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        torch.save(model.state_dict(),
                   os.path.join(save_model_path, 'HeteroTCR_epoch{:03d}_AUC{:.6f}.pth'.format(ep, val_ROCAUC.item())))

        loss.append(train_loss.item())
        acc.append(train_binary_accuracy.item())
        auc_roc.append(train_ROCAUC.item())
        precision.append(train_precision.item())
        recall.append(train_recall.item())
        f1_score.append(train_f1_score.item())
        specificity_list.append(train_specificity.item())

        val_loss.append(valid_loss.item())
        val_acc.append(val_binary_accuracy.item())
        val_auc_roc.append(val_ROCAUC.item())
        val_precision_list.append(val_precision_tensor.item())  # Ensure it is updated
        val_recall_list.append(val_recall_tensor.item())  # Ensure it is updated
        val_f1_score_list.append(val_f1_score_tensor.item())  # Ensure it is updated
        val_specificity_list.append(val_specificity.item())

        epoch.append(ep)

    # Logs of results
    dfhistory = {'epoch': epoch,
                 'loss': loss, 'acc': acc, 'auc_roc': auc_roc, 'precision': precision, 'recall': recall,
                 'f1_score': f1_score, 'specificity': specificity_list,
                 'val_loss': val_loss, 'val_acc': val_acc, 'val_auc_roc': val_auc_roc,
                 'val_precision': val_precision_list, 'val_recall': val_recall_list,
                 'val_f1_score': val_f1_score_list, 'val_specificity': val_specificity_list}  # Update the column name
   

    df = pd.DataFrame(dfhistory)
    if not os.path.exists(root_history):
        os.makedirs(root_history)
    df.to_csv(os.path.join(root_history, "HeteroTCR_{}.tsv".format(model_dir2)), header=True, sep='\t', index=False)

    # Get best epochs for AUC and Loss
    val_auc_roc = list(df['val_auc_roc'])
    max_auc_index = val_auc_roc.index(max(val_auc_roc))
    num_epoch = max_auc_index + 1
    print("max auc epoch:", num_epoch)
    print(max(val_auc_roc))

    val_loss = list(df['val_loss'])
    min_val_loss_index = val_loss.index(min(val_loss))
    loss_num_epoch = min_val_loss_index + 1
    print("loss auc epoch:", loss_num_epoch)
    print(val_auc_roc[min_val_loss_index])

    # Remove useless models
    for root, dirs, files in os.walk(save_model_path):
        for file in files:
            if file.startswith("HeteroTCR_epoch{:03d}".format(loss_num_epoch)):
                os.rename(os.path.join(save_model_path, file), os.path.join(save_model_path, 'minloss_AUC_' + file))
            elif file.startswith("HeteroTCR_epoch{:03d}".format(num_epoch)):
                os.rename(os.path.join(save_model_path, file), os.path.join(save_model_path, 'max_AUC_' + file))
            else:
                os.remove(os.path.join(save_model_path, file))
