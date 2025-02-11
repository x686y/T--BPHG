# import pandas as pd
#
# # 文件路径列表
# file_paths = [
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold0.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold1.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold2.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold3.tsv",
#     r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold4.tsv"
# ]
#
# # 用于存储所有肽及其预测概率的列表
# all_results = []
#
# # 逐个加载五折预测结果文件并提取肽、预测概率和真实标签
# for file in file_paths:
#     df = pd.read_csv(file, delimiter="\t")  # 假设是以制表符分隔
#     # 仅添加包含 peptide, probability, true_label 列的数据
#     all_results.append(df[['peptide', 'probability', 'true_label']])
#
# # 合并所有五折的结果
# merged_results = pd.concat(all_results, axis=0)
#
# # 仅保留正样本（true_label == 1）的数据
# positive_samples = merged_results[merged_results['true_label'] == 1]
#
# # 按肽名分组，并计算每个肽在正样本中的平均预测概率值
# average_probabilities_positive = positive_samples.groupby('peptide')['probability'].mean().reset_index()
#
# # 按肽名排序（可选）
# average_probabilities_positive = average_probabilities_positive.sort_values(by='probability', ascending=False)
#
# # 输出结果
# print(average_probabilities_positive)
import pandas as pd

# 文件路径列表
file_paths = [
    r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold0.tsv",
    r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold1.tsv",
    r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold2.tsv",
    r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold3.tsv",
    r"D:\xy\HeteroTCR-main\data\nettcr_bal5folds\predresult\nettcrbalHetero_pred_fold4.tsv"
]

# 用于存储所有肽及其预测概率的列表
all_results = []

# 逐个加载五折预测结果文件并提取肽、预测概率和真实标签
for file in file_paths:
    df = pd.read_csv(file, delimiter="\t")  # 假设是以制表符分隔
    # 仅添加包含 peptide, probability, true_label 列的数据
    all_results.append(df[['peptide', 'probability', 'true_label']])

# 合并所有五折的结果
merged_results = pd.concat(all_results, axis=0)

# 仅保留正样本（true_label == 1）的数据
positive_samples = merged_results[merged_results['true_label'] == 0]

# 按肽名分组，并计算每个肽在正样本中的平均预测概率值
average_probabilities_positive = positive_samples.groupby('peptide')['probability'].mean().reset_index()

# 按肽名排序（可选）
average_probabilities_positive = average_probabilities_positive.sort_values(by='probability', ascending=False)

# 输出结果
print(average_probabilities_positive)