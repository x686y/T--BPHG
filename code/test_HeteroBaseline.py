
import torch
import torch.nn.functional as F
import torchmetrics
import pandas as pd
import sys, os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from matplotlib import pyplot as plt

from scipy.stats import spearmanr
from sklearn.utils.class_weight import compute_class_weight
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from HeteroTCRAB_Modelnew4 import *
from Base_dataprogressNettcrfull import *
from config import *
# import matplotlib.pyplot as plt
# config
NUM_EPOCHS = int(args.epochs)
cuda_name = 'cuda:' + args.cuda
hc = int(args.hiddenchannels)
nl = int(args.numberlayers)
nt = args.gnnnet

root_model = args.modeldir

device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
data_test, test_edge_index_a, test_edge_index_b, y_test = create_dataset_global_predict()

# Initialize metrics for evaluation
test_precision_metric = torchmetrics.Precision(num_classes=2, average='macro', multiclass=True).to(device)
test_recall_metric = torchmetrics.Recall(num_classes=2, average='macro', multiclass=True).to(device)
test_f1_metric = torchmetrics.F1(num_classes=2, average='macro', multiclass=True).to(device)



def predictAB(model):
    # print('Testing on {} samples...'.format(len(data_test['cdr3b', 'binds_to', 'tra_peptide'].edge_index[0])))
    # print('Testing on {} samples...'.format(len(data_test['cdr3b', 'binds_to', 'trb_peptide'].edge_index[0])))
    loss_fn = torch.nn.BCELoss()

    model.eval()
    with torch.no_grad():
        outa_test, outb_test= model(data_test.x_dict, data_test.edge_index_dict,
                                                           test_edge_index_a, test_edge_index_b)
       
        out_test_tra_b = torch.stack([outa_test, outb_test], dim=-1)
        softmax_out_test_tra_b  = torch.softmax(out_test_tra_b, dim=-1)  
        test_weight_tra = softmax_out_test_tra_b[..., 0]
        test_weight_trb = softmax_out_test_tra_b[..., 1]
       
        print(f"Shape of test_weight_tra: {test_weight_tra.shape}")
        print(f"Shape of test_weight_trb: {test_weight_trb.shape}")
        out_test = (outa_test + outb_test) / 2 
        out_test =  out_test .squeeze(-1)
       
        out_test = torch.sigmoid(out_test)
        test_loss = loss_fn(out_test, torch.tensor(y_test).float().to(device))
        # test_loss = weighted_binary_crossentropy(torch.tensor(y_test).to(device), out_test)
        test_binary_accuracy = torchmetrics.functional.accuracy(out_test, torch.tensor(y_test).int().to(device))
        test_ROCAUC = torchmetrics.functional.auroc(out_test, torch.tensor(y_test).int().to(device))
        test_precision = test_precision_metric(out_test, torch.tensor(y_test).int().to(device))
        test_recall = test_recall_metric(out_test, torch.tensor(y_test).int().to(device))
        test_f1_score = test_f1_metric(out_test, torch.tensor(y_test).int().to(device))
    return test_weight_tra,test_weight_trb,test_loss.item(), test_binary_accuracy.item(), test_ROCAUC.item(), test_precision.item(), test_recall.item(), test_f1_score.item(), out_test

if __name__ == "__main__":
    model = HeteroTCR(data_test.metadata(), hc, nl, nt)
    model = model.to(device)
    data_test = data_test.to(device)

    with torch.no_grad():  # Initialize lazy modules.
        out_tra, out_trb= model(data_test.x_dict, data_test.edge_index_dict, test_edge_index_a, test_edge_index_b)

    model_dir = args.testmodeldir
    val_model_path = os.path.join(root_model, model_dir)



    for root, dirs, files in os.walk(val_model_path):
        for file in files:
            if file.startswith("max"):
                PATH = os.path.join(val_model_path, file)

                model.load_state_dict(torch.load(PATH))

                test_weight_tra,test_weight_trb,test_loss, test_binary_accuracy, test_ROCAUC, test_precision, test_recall, test_f1_score, test_prob = predictAB(model)
                print("ACC: {:.4f}".format(test_binary_accuracy))
                print("AUC: {:.4f}".format(test_ROCAUC))
                print("Precision: {:.4f}".format(test_precision))
                print("Recall: {:.4f}".format(test_recall))
                print("F1 Score: {:.4f}".format(test_f1_score))


                root = os.path.join(args.pridir, args.secdir, args.terdir)
                root2 = os.path.join(args.pridir, args.secdir, args.terdir2)
                test_data = pd.read_csv(os.path.join(root, 'test_data.tsv'), delimiter=args.testdelimiter)
                # After obtaining the test_prob from the prediction
                # test_labels = (test_prob >= 0.5).int()  # Convert probabilities to binary labels

                df = pd.DataFrame({
                    'tra_cdr3': test_data['A3'],
                    'trb_cdr3': test_data['B3'],
                    'peptide': test_data['peptide'],
                    'true_label': test_data['Binding'],
                    'probability': test_prob.cpu(),
                    # 'pre_lable': test_labels.cpu(),
                    'weights_tra': test_weight_tra.cpu().squeeze(), 
                    'weights_trb': test_weight_trb.cpu().squeeze() 

                })
                # df.to_csv(os.path.join(root2, "Base_nettcrfullHetero_pred_fold4.tsv"), header=True, sep='\t', index=False)
                # df.to_csv(os.path.join(root2, "Base_nettcrbalHetero_pred_fold4.tsv"), header=True, sep='\t', index=False)
                df.to_csv(os.path.join(root2, "Base_nettcr_strict4Hetero_pred_fold4.tsv"), header=True, sep='\t', index=False)

