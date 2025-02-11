# # 下面计算每一折的测试集每种肽与训练集的相似度，然后绘制AUC
# import pandas as pd
# import numpy as np
# from sklearn.metrics import roc_auc_score
# import Levenshtein
#
# # 读取数据
# train_data_path = "D:/xy/HeteroTCR-main/data/nettcr_strict0_5folds/fold0/train_data.tsv"
# test_data_path = "D:/xy/HeteroTCR-main/data/nettcr_strict0_5folds/fold0/test_data.tsv"
# pred_result_path = "D:/xy/HeteroTCR-main/data/nettcr_strict0_5folds/predresult/nettcr_strict0Hetero_pred_fold0.tsv"
#
# # 读取训练集、测试集、预测结果数据
# train_df = pd.read_csv(train_data_path, sep='\t')
# test_df = pd.read_csv(test_data_path, sep='\t')
# pred_df = pd.read_csv(pred_result_path, sep='\t')
#
# # 提取肽（peptide）列和标签列（Binding/true_label）
# train_peptides = train_df['peptide'].values
# test_peptides = test_df['peptide'].values
# test_labels = test_df['Binding'].values
# pred_probs = pred_df['probability'].values
#
# # 计算最小Levenshtein距离
# def compute_min_levenshtein(test_peptide, train_peptides):
#     distances = [Levenshtein.distance(test_peptide, train_peptide) for train_peptide in train_peptides]
#     return min(distances)
#
# # 计算每个测试集肽的最小Levenshtein距离
# lmin_values = [compute_min_levenshtein(test_peptide, train_peptides) for test_peptide in test_peptides]
#
# # 计算每种肽的平均AUC值
# # 我们需要先对test_peptides进行分组，计算每个肽的平均AUC值
# unique_peptides = np.unique(test_peptides)  # 获取所有独特的肽
# avg_auc_per_peptide = {}
#
# for peptide in unique_peptides:
#     # 获取该肽对应的所有测试样本的索引
#     peptide_indices = np.where(test_peptides == peptide)[0]
#     # 获取这些样本的真实标签和预测概率
#     peptide_labels = test_labels[peptide_indices]
#     peptide_probs = pred_probs[peptide_indices]
#     # 计算该肽对应样本的平均AUC
#     avg_auc = roc_auc_score(peptide_labels, peptide_probs)
#     avg_auc_per_peptide[peptide] = avg_auc
#
# # 打印每个肽的平均AUC值和对应的Levenshtein距离
# print("Test Peptide | Lmin (Levenshtein Distance) | Average ROC-AUC")
# for peptide in unique_peptides:
#     # 获取该肽的最小Levenshtein距离
#     lmin = compute_min_levenshtein(peptide, train_peptides)
#     avg_auc = avg_auc_per_peptide[peptide]
#     print(f"{peptide} | {lmin} | {avg_auc}")
#
# # 如果需要按Lmin值排序
# sorted_peptides = sorted(unique_peptides, key=lambda peptide: compute_min_levenshtein(peptide, train_peptides))
# print("\nSorted by Lmin:")
# print("Test Peptide | Lmin (Levenshtein Distance) | Average ROC-AUC")
# for peptide in sorted_peptides:
#     lmin = compute_min_levenshtein(peptide, train_peptides)
#     avg_auc = avg_auc_per_peptide[peptide]
#     print(f"{peptide} | {lmin} | {avg_auc}")
# =================================
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
import Levenshtein

# 读取数据
train_data_path = "D:/xy/HeteroTCR-main/data/nettcr_strict0_5folds/fold4/train_data.tsv"
test_data_path = "D:/xy/HeteroTCR-main/data/nettcr_strict0_5folds/fold4/test_data.tsv"
pred_result_path = "D:/xy/HeteroTCR-main/data/nettcr_strict0_5folds/predresult/nettcr_strict0Hetero_pred_fold4.tsv"

# 读取训练集、测试集、预测结果数据
train_df = pd.read_csv(train_data_path, sep='\t')
test_df = pd.read_csv(test_data_path, sep='\t')
pred_df = pd.read_csv(pred_result_path, sep='\t')

# 提取肽（peptide）列和标签列（Binding/true_label）
train_peptides = train_df['peptide'].values
test_peptides = test_df['peptide'].values
test_labels = test_df['Binding'].values
pred_probs = pred_df['probability'].values

# 计算最小Levenshtein距离
def compute_min_levenshtein(test_peptide, train_peptides):
    distances = [Levenshtein.distance(test_peptide, train_peptide) for train_peptide in train_peptides]
    return min(distances)

# 计算每个测试集肽的最小Levenshtein距离
lmin_values = [compute_min_levenshtein(test_peptide, train_peptides) for test_peptide in test_peptides]

# 计算每种肽的平均AUPRC值
# 我们需要先对test_peptides进行分组，计算每个肽的平均AUPRC值
unique_peptides = np.unique(test_peptides)  # 获取所有独特的肽
avg_auprc_per_peptide = {}

for peptide in unique_peptides:
    # 获取该肽对应的所有测试样本的索引
    peptide_indices = np.where(test_peptides == peptide)[0]
    # 获取这些样本的真实标签和预测概率
    peptide_labels = test_labels[peptide_indices]
    peptide_probs = pred_probs[peptide_indices]
    # 计算该肽对应样本的平均AUPRC
    avg_auprc = average_precision_score(peptide_labels, peptide_probs)
    avg_auprc_per_peptide[peptide] = avg_auprc

# 打印每个肽的平均AUPRC值和对应的Levenshtein距离
print("Test Peptide | Lmin (Levenshtein Distance) | Average AUPRC")
for peptide in unique_peptides:
    # 获取该肽的最小Levenshtein距离
    lmin = compute_min_levenshtein(peptide, train_peptides)
    avg_auprc = avg_auprc_per_peptide[peptide]
    print(f"{peptide} | {lmin} | {avg_auprc}")

# 如果需要按Lmin值排序
sorted_peptides = sorted(unique_peptides, key=lambda peptide: compute_min_levenshtein(peptide, train_peptides))
print("\nSorted by Lmin:")
print("Test Peptide | Lmin (Levenshtein Distance) | Average AUPRC")
for peptide in sorted_peptides:
    lmin = compute_min_levenshtein(peptide, train_peptides)
    avg_auprc = avg_auprc_per_peptide[peptide]
    print(f"{peptide} | {lmin} | {avg_auprc}")

