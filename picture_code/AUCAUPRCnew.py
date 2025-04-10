# 
# import pandas as pd
# import numpy as np
# from sklearn.metrics import average_precision_score
#
# 
# # files = [
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold4.tsv"
# # ]
# files = [
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold4.tsv"
# ]
# # 
# all_data = []
#
# # 
# for fold, file in enumerate(files):
#     df = pd.read_csv(file, sep='\t')
#     df['fold'] = fold  
#     all_data.append(df)
#
# 
# all_df = pd.concat(all_data, ignore_index=True)
#
# 
# peptide_auprc = {}
#
# 
# peptides = all_df['peptide'].unique()
#
# 
# for peptide in peptides:
#    
#     peptide_data = all_df[all_df['peptide'] == peptide]
#
#   
#     sample_count = len(peptide_data)
#
#     # Initialize a list to store the five-fold AUPRC values of the peptide.
#     fold_auprc = []
#
#     
#     for fold in range(5):
#         fold_data = peptide_data[peptide_data['fold'] == fold]
#         y_true = fold_data['true_label']
#         y_prob = fold_data['probability']
#
#        
#         if len(y_true) > 0 and len(y_prob) > 0:
#             try:
#                 auprc = average_precision_score(y_true, y_prob)
#                 fold_auprc.append(auprc)
#             except ValueError as e:
#                 print(f"Error computing AUPRC for peptide {peptide}, fold {fold}: {e}")
#                 continue
#
#     # Calculate the average AUPRC value and standard deviation of the peptide.
#     if fold_auprc:
#         avg_auprc = np.mean(fold_auprc)
#         std_auprc = np.std(fold_auprc)
#         peptide_auprc[peptide] = {'average_auprc': avg_auprc, 'std_auprc': std_auprc, 'sample_count': sample_count}
#
#
# sorted_peptides = sorted(peptide_auprc.items(), key=lambda x: x[1]['sample_count'], reverse=True)
#
# 
# for peptide, metrics in sorted_peptides:
#     print(f"Peptide: {peptide}")
#     print(f"Sample Count: {metrics['sample_count']}")
#     print(f"Average AUPRC: {metrics['average_auprc']:.4f}")
#     print(f"AUPRC Standard Deviation: {metrics['std_auprc']:.4f}")
#     print('-' * 50)
#
# #Calculate and output the average AUC for each peptide, sorted by the sample size of each peptide in descending order.
# import pandas as pd
# import numpy as np
# from sklearn.metrics import roc_auc_score
#
# 
# # files = [
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold4.tsv"
# # ]
# files = [
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold4.tsv"
# ]
# 
# all_data = []
#
# 
# for fold, file in enumerate(files):
#     df = pd.read_csv(file, sep='\t')
#     df['fold'] = fold 
#     all_data.append(df)
#
# 
# all_df = pd.concat(all_data, ignore_index=True)
#
# 
# peptide_auroc = {}
#
#
# peptides = all_df['peptide'].unique()
#
# 
# for peptide in peptides:
#   
#     peptide_data = all_df[all_df['peptide'] == peptide]
#
#    
#     sample_count = len(peptide_data)
#
#    
#     fold_auroc = []
#
#     
#     for fold in range(5):
#         fold_data = peptide_data[peptide_data['fold'] == fold]
#         y_true = fold_data['true_label']
#         y_prob = fold_data['probability']
#
#        
#         if len(y_true) > 0 and len(y_prob) > 0:
#             try:
#                 auroc = roc_auc_score(y_true, y_prob)
#                 fold_auroc.append(auroc)
#             except ValueError as e:
#                 print(f"Error computing AUROC for peptide {peptide}, fold {fold}: {e}")
#                 continue
#
#    
#     if fold_auroc:
#         avg_auroc = np.mean(fold_auroc)
#         std_auroc = np.std(fold_auroc)
#         peptide_auroc[peptide] = {'average_auroc': avg_auroc, 'std_auroc': std_auroc, 'sample_count': sample_count}
#
# 
# sorted_peptides = sorted(peptide_auroc.items(), key=lambda x: x[1]['sample_count'], reverse=True)
#
# 
# for peptide, metrics in sorted_peptides:
#     print(f"Peptide: {peptide}")
#     print(f"Sample Count: {metrics['sample_count']}")
#     print(f"Average AUROC: {metrics['average_auroc']:.4f}")
#     print(f"AUROC Standard Deviation: {metrics['std_auroc']:.4f}")
#     print('-' * 50)

# Calculate the overall 5-fold.
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# import numpy as np
# 
# # file_paths = [
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0Hetero_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0Hetero_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0Hetero_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0Hetero_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0Hetero_pred_fold4.tsv"
# # ]
# # file_paths = [
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\nettcr_strict4Hetero_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\nettcr_strict4Hetero_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\nettcr_strict4Hetero_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\nettcr_strict4Hetero_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\nettcr_strict4Hetero_pred_fold4.tsv"
# # ]
# # file_paths = [
# #     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\PaperNettcrbal_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\PaperNettcrbal_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\PaperNettcrbal_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\PaperNettcrbal_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\PaperNettcrbal_pred_fold4.tsv"
# # ]
# # file_paths = [
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\PaperNettcrfull_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\PaperNettcrfull_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\PaperNettcrfull_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\PaperNettcrfull_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\PaperNettcrfull_pred_fold4.tsv"
# # ]
# # file_paths = [
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\PaperNettcrStrict0_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\PaperNettcrStrict0_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\PaperNettcrStrict0_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\PaperNettcrStrict0_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\PaperNettcrStrict0_pred_fold4.tsv"
# # ]
# # file_paths = [
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\PaperNettcrStrict1_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\PaperNettcrStrict1_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\PaperNettcrStrict1_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\PaperNettcrStrict1_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\PaperNettcrStrict1_pred_fold4.tsv"
# # ]
# # file_paths = [
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\PaperNettcrStrict2_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\PaperNettcrStrict2_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\PaperNettcrStrict2_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\PaperNettcrStrict2_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\PaperNettcrStrict2_pred_fold4.tsv"
# # ]
# # file_paths = [
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\PaperNettcrStrict3_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\PaperNettcrStrict3_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\PaperNettcrStrict3_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\PaperNettcrStrict3_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\PaperNettcrStrict3_pred_fold4.tsv"
# # ]
# # file_paths = [
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\PaperNettcrStrict4_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\PaperNettcrStrict4_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\PaperNettcrStrict4_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\PaperNettcrStrict4_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\PaperNettcrStrict4_pred_fold4.tsv"
# # ]
# # 
# # file_paths = [
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold4.tsv"
# # ]
# # 
# # file_paths = [
# #     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold4.tsv"
# # ]
#
# # file_paths = [
# #     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\Base_nettcrbalHetero_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\Base_nettcrbalHetero_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\Base_nettcrbalHetero_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\Base_nettcrbalHetero_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\Base_nettcrbalHetero_pred_fold4.tsv"
# # ]
# # file_paths = [
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\Base_nettcrfullHetero_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\Base_nettcrfullHetero_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\Base_nettcrfullHetero_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\Base_nettcrfullHetero_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\Base_nettcrfullHetero_pred_fold4.tsv"
# # ]
# # file_paths = [
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\Base_nettcr_strict0Hetero_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\Base_nettcr_strict0Hetero_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\Base_nettcr_strict0Hetero_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\Base_nettcr_strict0Hetero_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\Base_nettcr_strict0Hetero_pred_fold4.tsv"
# # ]
# # file_paths = [
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\Base_nettcr_strict1Hetero_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\Base_nettcr_strict1Hetero_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\Base_nettcr_strict1Hetero_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\Base_nettcr_strict1Hetero_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\Base_nettcr_strict1Hetero_pred_fold4.tsv"
# # ]
# # file_paths = [
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\Base_nettcr_strict2Hetero_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\Base_nettcr_strict2Hetero_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\Base_nettcr_strict2Hetero_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\Base_nettcr_strict2Hetero_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\Base_nettcr_strict2Hetero_pred_fold4.tsv"
# # ]
# # file_paths = [
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\Base_nettcr_strict3Hetero_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\Base_nettcr_strict3Hetero_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\Base_nettcr_strict3Hetero_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\Base_nettcr_strict3Hetero_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\Base_nettcr_strict3Hetero_pred_fold4.tsv"
# # ]
# # file_paths = [
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\Base_nettcr_strict4Hetero_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\Base_nettcr_strict4Hetero_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\Base_nettcr_strict4Hetero_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\Base_nettcr_strict4Hetero_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\Base_nettcr_strict4Hetero_pred_fold4.tsv"
# # ]
# # file_paths = [
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullNoGNN_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullNoGNN_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullNoGNN_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullNoGNN_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullNoGNN_pred_fold4.tsv"
# # ]
# # file_paths = [
# #     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcr_balNoGNN_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcr_balNoGNN_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcr_balNoGNN_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcr_balNoGNN_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcr_balNoGNN_pred_fold4.tsv"
# # ]
# # file_paths = [
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0_5foldsNoGNN_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0_5foldsNoGNN_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0_5foldsNoGNN_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0_5foldsNoGNN_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0_5foldsNoGNN_pred_fold4.tsv"
# # ]
# # file_paths = [
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\nettcr_strict1_5foldsNoGNN_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\nettcr_strict1_5foldsNoGNN_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\nettcr_strict1_5foldsNoGNN_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\nettcr_strict1_5foldsNoGNN_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\nettcr_strict1_5foldsNoGNN_pred_fold4.tsv"
# # ]
# # file_paths = [
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\nettcr_strict2_5foldsNoGNN_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\nettcr_strict2_5foldsNoGNN_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\nettcr_strict2_5foldsNoGNN_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\nettcr_strict2_5foldsNoGNN_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\nettcr_strict2_5foldsNoGNN_pred_fold4.tsv"
# # ]
# # file_paths = [
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\nettcr_strict3_5foldsNoGNN_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\nettcr_strict3_5foldsNoGNN_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\nettcr_strict3_5foldsNoGNN_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\nettcr_strict3_5foldsNoGNN_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\nettcr_strict3_5foldsNoGNN_pred_fold4.tsv"
# # ]
# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\nettcr_strict4_5foldsNoGNN_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\nettcr_strict4_5foldsNoGNN_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\nettcr_strict4_5foldsNoGNN_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\nettcr_strict4_5foldsNoGNN_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\nettcr_strict4_5foldsNoGNN_pred_fold4.tsv"
# ]
# 
# fold_aucs = []
#
# 
# plt.figure(figsize=(10, 8))
#
# 
# for i, file_path in enumerate(file_paths):
#     data = pd.read_csv(file_path, sep='\t')
#     true_labels = data['true_label']
#     pred_scores = data['probability']
#
#    
#     fpr, tpr, _ = roc_curve(true_labels, pred_scores)
#     roc_auc = auc(fpr, tpr)
#     fold_aucs.append(roc_auc)
#
#    
#     plt.plot(fpr, tpr, label=f'Fold {i + 1} (AUC = {roc_auc:.3f})')
#
# C
# mean_auc = np.mean(fold_aucs)
# 
# std_auc = np.std(fold_aucs)
#
# 
# plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Guess')
#
# 
# plt.title(f'ROC Curves for Five Folds (Mean AUC = {mean_auc:.3f}, Std = {std_auc:.3f})', fontsize=16)
# plt.xlabel('False Positive Rate', fontsize=12)
# plt.ylabel('True Positive Rate', fontsize=12)
# plt.legend(loc='lower right', fontsize=10)
# plt.grid(alpha=0.5)
# plt.tight_layout()
# plt.show()
#
# 
# for i, auc_value in enumerate(fold_aucs):
#     print(f'Fold {i + 1}: AUC = {auc_value:.3f}')
# print(f'Mean AUC across all folds: {mean_auc:.3f}')
# print(f'Standard Deviation of AUC across folds: {std_auc:.3f}')

#Calculate the five-fold.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import numpy as np

# 
# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold4.tsv"
# ]
# 
# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold4.tsv"
# ]
# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0Hetero_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0Hetero_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0Hetero_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0Hetero_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0Hetero_pred_fold4.tsv"
# ]
# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\PaperNettcrbal_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\PaperNettcrbal_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\PaperNettcrbal_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\PaperNettcrbal_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\PaperNettcrbal_pred_fold4.tsv"
# ]
# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\PaperNettcrfull_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\PaperNettcrfull_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\PaperNettcrfull_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\PaperNettcrfull_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\PaperNettcrfull_pred_fold4.tsv"
# ]
# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0Hetero_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0Hetero_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0Hetero_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0Hetero_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0Hetero_pred_fold4.tsv"
# ]
# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\PaperNettcrStrict1_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\PaperNettcrStrict1_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\PaperNettcrStrict1_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\PaperNettcrStrict1_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\PaperNettcrStrict1_pred_fold4.tsv"
# ]
# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\PaperNettcrStrict2_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\PaperNettcrStrict2_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\PaperNettcrStrict2_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\PaperNettcrStrict2_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\PaperNettcrStrict2_pred_fold4.tsv"
# ]
# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\PaperNettcrStrict3_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\PaperNettcrStrict3_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\PaperNettcrStrict3_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\PaperNettcrStrict3_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\PaperNettcrStrict3_pred_fold4.tsv"
# ]
# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\PaperNettcrStrict4_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\PaperNettcrStrict4_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\PaperNettcrStrict4_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\PaperNettcrStrict4_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\PaperNettcrStrict4_pred_fold4.tsv"
# ]

# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\Base_nettcrbalHetero_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\Base_nettcrbalHetero_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\Base_nettcrbalHetero_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\Base_nettcrbalHetero_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\Base_nettcrbalHetero_pred_fold4.tsv"
# ]
# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\Base_nettcrfullHetero_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\Base_nettcrfullHetero_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\Base_nettcrfullHetero_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\Base_nettcrfullHetero_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\Base_nettcrfullHetero_pred_fold4.tsv"
# ]
# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\Base_nettcr_strict0Hetero_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\Base_nettcr_strict0Hetero_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\Base_nettcr_strict0Hetero_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\Base_nettcr_strict0Hetero_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\Base_nettcr_strict0Hetero_pred_fold4.tsv"
# ]
# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\Base_nettcr_strict1Hetero_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\Base_nettcr_strict1Hetero_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\Base_nettcr_strict1Hetero_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\Base_nettcr_strict1Hetero_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\Base_nettcr_strict1Hetero_pred_fold4.tsv"
# ]
# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\Base_nettcr_strict2Hetero_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\Base_nettcr_strict2Hetero_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\Base_nettcr_strict2Hetero_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\Base_nettcr_strict2Hetero_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\Base_nettcr_strict2Hetero_pred_fold4.tsv"
# ]
# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\Base_nettcr_strict3Hetero_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\Base_nettcr_strict3Hetero_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\Base_nettcr_strict3Hetero_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\Base_nettcr_strict3Hetero_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\Base_nettcr_strict3Hetero_pred_fold4.tsv"
# ]
# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\Base_nettcr_strict4Hetero_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\Base_nettcr_strict4Hetero_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\Base_nettcr_strict4Hetero_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\Base_nettcr_strict4Hetero_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\Base_nettcr_strict4Hetero_pred_fold4.tsv"
# ]

# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullNoGNN_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullNoGNN_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullNoGNN_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullNoGNN_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullNoGNN_pred_fold4.tsv"
# ]
# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcr_balNoGNN_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcr_balNoGNN_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcr_balNoGNN_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcr_balNoGNN_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcr_balNoGNN_pred_fold4.tsv"
# ]
# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0_5foldsNoGNN_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0_5foldsNoGNN_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0_5foldsNoGNN_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0_5foldsNoGNN_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0_5foldsNoGNN_pred_fold4.tsv"
# ]
# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\nettcr_strict1_5foldsNoGNN_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\nettcr_strict1_5foldsNoGNN_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\nettcr_strict1_5foldsNoGNN_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\nettcr_strict1_5foldsNoGNN_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict1_5folds\predresult\nettcr_strict1_5foldsNoGNN_pred_fold4.tsv"
# ]
# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\nettcr_strict2_5foldsNoGNN_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\nettcr_strict2_5foldsNoGNN_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\nettcr_strict2_5foldsNoGNN_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\nettcr_strict2_5foldsNoGNN_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict2_5folds\predresult\nettcr_strict2_5foldsNoGNN_pred_fold4.tsv"
# ]
# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\nettcr_strict3_5foldsNoGNN_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\nettcr_strict3_5foldsNoGNN_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\nettcr_strict3_5foldsNoGNN_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\nettcr_strict3_5foldsNoGNN_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_strict3_5folds\predresult\nettcr_strict3_5foldsNoGNN_pred_fold4.tsv"
# ]
file_paths = [
    r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\nettcr_strict4_5foldsNoGNN_pred_fold0.tsv",
    r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\nettcr_strict4_5foldsNoGNN_pred_fold1.tsv",
    r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\nettcr_strict4_5foldsNoGNN_pred_fold2.tsv",
    r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\nettcr_strict4_5foldsNoGNN_pred_fold3.tsv",
    r"D:\xy\HeteroTCR-main\data\nettcr_strict4_5folds\predresult\nettcr_strict4_5foldsNoGNN_pred_fold4.tsv"
]
#
fold_auprcs = []


plt.figure(figsize=(10, 8))


for i, file_path in enumerate(file_paths):
    data = pd.read_csv(file_path, sep='\t')
    true_labels = data['true_label']
    pred_scores = data['probability']

    
    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    auprc = auc(recall, precision)
    fold_auprcs.append(auprc)

   
    plt.plot(recall, precision, label=f'Fold {i + 1} (AUPRC = {auprc:.3f})')


mean_auprc = np.mean(fold_auprcs)

std_auprc = np.std(fold_auprcs)


plt.plot([0, 1], [0.5, 0.5], 'k--', lw=1, label='Random Guess')


plt.title(f'Precision-Recall Curves for Five Folds (Mean AUPRC = {mean_auprc:.3f}, Std = {std_auprc:.3f})', fontsize=16)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.legend(loc='lower left', fontsize=10)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()


for i, auprc_value in enumerate(fold_auprcs):
    print(f'Fold {i + 1}: AUPRC = {auprc_value:.3f}')
print(f'Mean AUPRC across all folds: {mean_auprc:.3f}')
print(f'Standard Deviation of AUPRC across folds: {std_auprc:.3f}')
