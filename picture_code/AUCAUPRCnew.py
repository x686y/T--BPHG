# 下面是计算按每种肽的样本数量从大到小输出每种肽的平均AUPRC
# import pandas as pd
# import numpy as np
# from sklearn.metrics import average_precision_score
#
# # 读取五个折的文件路径
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
# # 初始化存储所有文件数据的列表
# all_data = []
#
# # 读取每个文件并加入到列表中，推断每个文件对应的折号
# for fold, file in enumerate(files):
#     df = pd.read_csv(file, sep='\t')
#     df['fold'] = fold  # 添加一个'fold'列，表示该数据对应的折号
#     all_data.append(df)
#
# # 合并所有文件的数据
# all_df = pd.concat(all_data, ignore_index=True)
#
# # 初始化一个字典来存储每个肽的AUPRC值和样本数
# peptide_auprc = {}
#
# # 获取所有独特的肽名
# peptides = all_df['peptide'].unique()
#
# # 对每种肽计算五折的AUPRC值及样本数
# for peptide in peptides:
#     # 获取该肽的所有预测结果
#     peptide_data = all_df[all_df['peptide'] == peptide]
#
#     # 统计该肽的样本数
#     sample_count = len(peptide_data)
#
#     # 初始化一个列表来存储该肽的五折AUPRC值
#     fold_auprc = []
#
#     # 按折号（fold）计算AUPRC
#     for fold in range(5):
#         fold_data = peptide_data[peptide_data['fold'] == fold]
#         y_true = fold_data['true_label']
#         y_prob = fold_data['probability']
#
#         # 确保fold数据非空
#         if len(y_true) > 0 and len(y_prob) > 0:
#             try:
#                 auprc = average_precision_score(y_true, y_prob)
#                 fold_auprc.append(auprc)
#             except ValueError as e:
#                 print(f"Error computing AUPRC for peptide {peptide}, fold {fold}: {e}")
#                 continue
#
#     # 计算该肽的平均AUPRC值和标准差
#     if fold_auprc:
#         avg_auprc = np.mean(fold_auprc)
#         std_auprc = np.std(fold_auprc)
#         peptide_auprc[peptide] = {'average_auprc': avg_auprc, 'std_auprc': std_auprc, 'sample_count': sample_count}
#
# # 根据样本数从大到小排序
# sorted_peptides = sorted(peptide_auprc.items(), key=lambda x: x[1]['sample_count'], reverse=True)
#
# # 打印每种肽的AUPRC平均值、标准差以及样本数
# for peptide, metrics in sorted_peptides:
#     print(f"Peptide: {peptide}")
#     print(f"Sample Count: {metrics['sample_count']}")
#     print(f"Average AUPRC: {metrics['average_auprc']:.4f}")
#     print(f"AUPRC Standard Deviation: {metrics['std_auprc']:.4f}")
#     print('-' * 50)
# ==============================================
# # 下面是计算按每种肽的样本数量从大到小输出每种肽的平均AUC
# import pandas as pd
# import numpy as np
# from sklearn.metrics import roc_auc_score
#
# # 读取五个折的文件路径
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
# # 初始化存储所有文件数据的列表
# all_data = []
#
# # 读取每个文件并加入到列表中，推断每个文件对应的折号
# for fold, file in enumerate(files):
#     df = pd.read_csv(file, sep='\t')
#     df['fold'] = fold  # 添加一个'fold'列，表示该数据对应的折号
#     all_data.append(df)
#
# # 合并所有文件的数据
# all_df = pd.concat(all_data, ignore_index=True)
#
# # 初始化一个字典来存储每个肽的AUROC值和样本数
# peptide_auroc = {}
#
# # 获取所有独特的肽名
# peptides = all_df['peptide'].unique()
#
# # 对每种肽计算五折的AUROC值及样本数
# for peptide in peptides:
#     # 获取该肽的所有预测结果
#     peptide_data = all_df[all_df['peptide'] == peptide]
#
#     # 统计该肽的样本数
#     sample_count = len(peptide_data)
#
#     # 初始化一个列表来存储该肽的五折AUROC值
#     fold_auroc = []
#
#     # 按折号（fold）计算AUROC
#     for fold in range(5):
#         fold_data = peptide_data[peptide_data['fold'] == fold]
#         y_true = fold_data['true_label']
#         y_prob = fold_data['probability']
#
#         # 确保fold数据非空
#         if len(y_true) > 0 and len(y_prob) > 0:
#             try:
#                 auroc = roc_auc_score(y_true, y_prob)
#                 fold_auroc.append(auroc)
#             except ValueError as e:
#                 print(f"Error computing AUROC for peptide {peptide}, fold {fold}: {e}")
#                 continue
#
#     # 计算该肽的平均AUROC值和标准差
#     if fold_auroc:
#         avg_auroc = np.mean(fold_auroc)
#         std_auroc = np.std(fold_auroc)
#         peptide_auroc[peptide] = {'average_auroc': avg_auroc, 'std_auroc': std_auroc, 'sample_count': sample_count}
#
# # 根据样本数从大到小排序
# sorted_peptides = sorted(peptide_auroc.items(), key=lambda x: x[1]['sample_count'], reverse=True)
#
# # 打印每种肽的AUROC平均值、标准差以及样本数
# for peptide, metrics in sorted_peptides:
#     print(f"Peptide: {peptide}")
#     print(f"Sample Count: {metrics['sample_count']}")
#     print(f"Average AUROC: {metrics['average_auroc']:.4f}")
#     print(f"AUROC Standard Deviation: {metrics['std_auroc']:.4f}")
#     print('-' * 50)

# # # =======================================================计算整体5折AUROC
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# import numpy as np
# # 文件路径列表
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
# # # 文件路径列表
# # file_paths = [
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold0.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold1.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold2.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold3.tsv",
# #     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold4.tsv"
# # ]
# # # 文件路径列表
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
# # 存储每折的AUC值
# fold_aucs = []
#
# # 创建图形
# plt.figure(figsize=(10, 8))
#
# # 遍历每个文件，计算ROC曲线和AUC
# for i, file_path in enumerate(file_paths):
#     data = pd.read_csv(file_path, sep='\t')
#     true_labels = data['true_label']
#     pred_scores = data['probability']
#
#     # 计算ROC曲线
#     fpr, tpr, _ = roc_curve(true_labels, pred_scores)
#     roc_auc = auc(fpr, tpr)
#     fold_aucs.append(roc_auc)
#
#     # 绘制每折的ROC曲线
#     plt.plot(fpr, tpr, label=f'Fold {i + 1} (AUC = {roc_auc:.3f})')
#
# # 计算平均AUROC
# mean_auc = np.mean(fold_aucs)
# # 计算标准差
# std_auc = np.std(fold_aucs)
#
# # 绘制随机猜测的参考线
# plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Guess')
#
# # 图形配置
# plt.title(f'ROC Curves for Five Folds (Mean AUC = {mean_auc:.3f}, Std = {std_auc:.3f})', fontsize=16)
# plt.xlabel('False Positive Rate', fontsize=12)
# plt.ylabel('True Positive Rate', fontsize=12)
# plt.legend(loc='lower right', fontsize=10)
# plt.grid(alpha=0.5)
# plt.tight_layout()
# plt.show()
#
# # 输出每折AUC值、平均AUC值和标准差
# for i, auc_value in enumerate(fold_aucs):
#     print(f'Fold {i + 1}: AUC = {auc_value:.3f}')
# print(f'Mean AUC across all folds: {mean_auc:.3f}')
# print(f'Standard Deviation of AUC across folds: {std_auc:.3f}')

# =======================================================计算五折AUPRC
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import numpy as np

# 文件路径列表
# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcrfull5folds\predresult\nettcrfullHetero_pred_fold4.tsv"
# ]
# # 文件路径列表
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
# 存储每折的AUPRC值
fold_auprcs = []

# 创建图形
plt.figure(figsize=(10, 8))

# 遍历每个文件，计算PR曲线和AUPRC
for i, file_path in enumerate(file_paths):
    data = pd.read_csv(file_path, sep='\t')
    true_labels = data['true_label']
    pred_scores = data['probability']

    # 计算PR曲线
    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    auprc = auc(recall, precision)
    fold_auprcs.append(auprc)

    # 绘制每折的PR曲线
    plt.plot(recall, precision, label=f'Fold {i + 1} (AUPRC = {auprc:.3f})')

# 计算平均AUPRC
mean_auprc = np.mean(fold_auprcs)
# 计算标准差
std_auprc = np.std(fold_auprcs)

# 绘制随机猜测的参考线（即精度=召回率）
plt.plot([0, 1], [0.5, 0.5], 'k--', lw=1, label='Random Guess')

# 图形配置
plt.title(f'Precision-Recall Curves for Five Folds (Mean AUPRC = {mean_auprc:.3f}, Std = {std_auprc:.3f})', fontsize=16)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.legend(loc='lower left', fontsize=10)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

# 输出每折AUPRC值、平均AUPRC值和标准差
for i, auprc_value in enumerate(fold_auprcs):
    print(f'Fold {i + 1}: AUPRC = {auprc_value:.3f}')
print(f'Mean AUPRC across all folds: {mean_auprc:.3f}')
print(f'Standard Deviation of AUPRC across folds: {std_auprc:.3f}')
