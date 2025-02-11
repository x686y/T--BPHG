# import pandas as pd
#
# # 读取数据
# file_path = r'D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0Hetero_pred_fold0.tsv'
# df = pd.read_csv(file_path, sep='\t')
#
# # 筛选符合条件的样本：peptide列为 'LLWNGPMAV'，true_label列为1，probability列大于0.5
# filtered_df = df[(df['peptide'] == 'LLWNGPMAV') &
#                  (df['true_label'] == 1) &
#                  (df['probability'] > 0.5)]
#
# # 计算weights_tra列和weights_trb列的平均值
# mean_weights_tra = filtered_df['weights_tra'].mean()
# mean_weights_trb = filtered_df['weights_trb'].mean()
#
# # 打印结果
# print(f"平均值 of weights_tra: {mean_weights_tra}")
# print(f"平均值 of weights_trb: {mean_weights_trb}")
# ==================================每个肽的
# import pandas as pd
#
# # 读取数据
# file_path = r'D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0Hetero_pred_fold4.tsv'
# df = pd.read_csv(file_path, sep='\t')
#
# # 筛选符合条件的样本：true_label列为1，probability列大于0.5
# filtered_df = df[(df['true_label'] == 1) & (df['probability'] > 0.5)]
#
# # 按照 peptide 列分组，并计算每个 peptide 的 weights_tra 和 weights_trb 的平均值
# mean_values = filtered_df.groupby('peptide')[['weights_tra', 'weights_trb']].mean()
#
# # 打印结果
# print("每种 peptide 对应的 weights_tra 和 weights_trb 平均值：")
# print(mean_values)

# ============================netstrict0_fold0正样本置信度曲线
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 定义要分析的文件路径
file_paths = [
    r'D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0Hetero_pred_fold0.tsv',
    r'D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0Hetero_pred_fold1.tsv',
    r'D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0Hetero_pred_fold3.tsv'
]

# 定义肽及其对应的条件
peptides_conditions = {
    'LLWNGPMAV': {
        'file_index': 0,  # 对应第一个文件
        'peptide': 'LLWNGPMAV'
    },
    'GILGFVFTL': {
        'file_index': 0,  # 对应第一个文件
        'peptide': 'GILGFVFTL'
    },
    'IVTDFSVIK': {
        'file_index': 1,  # 对应第二个文件
        'peptide': 'IVTDFSVIK'
    },
    'NLVPMVATV': {
        'file_index': 2,  # 对应第三个文件
        'peptide': 'NLVPMVATV'
    }
}

# 初始化字典存储每个肽的正样本概率
peptide_probabilities = {}

# 筛选每个肽的正样本并存储概率
for peptide, condition in peptides_conditions.items():
    file_path = file_paths[condition['file_index']]

    # 读取数据
    df = pd.read_csv(file_path, sep='\t')

    # 筛选正样本条件：true_label == 1 和 probability > 0.5
    filtered_df = df[
        (df['peptide'] == condition['peptide']) &
        (df['true_label'] == 1) &
        (df['probability'] > 0.5)
        ]

    if filtered_df.empty:
        print(f"警告: 在文件 {file_path} 中未找到肽 {peptide} 的正样本 (probability > 0.5)。")
    else:
        peptide_probabilities[peptide] = filtered_df['probability'].values

# 定义图例颜色（根据需要进行调整）
legend_colors = {
    'LLWNGPMAV': (91 / 255, 191 / 255, 192 / 255),
    'GILGFVFTL': (149 / 255, 213 / 255, 184 / 255),
    'IVTDFSVIK': (255 / 255, 123 / 255, 115 / 255),  # 添加IVTDFSVIK的颜色
    'NLVPMVATV': (255 / 255, 163 / 255, 67 / 255)  # 添加ELAGIGILTV的颜色
}

# 绘制密度分布图
plt.figure(figsize=(8, 6))

for peptide, probabilities in peptide_probabilities.items():
    sns.kdeplot(probabilities, label=peptide, fill=True, alpha=0.5, color=legend_colors.get(peptide, None), linewidth=2,
                edgecolor=legend_colors.get(peptide, None))

# 添加轴标签和标题
plt.xlabel("Confidence Score", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.title("Confidence Scores for Positive Predictions by Peptide", fontsize=14)

# 在左上角标注字母"A"
plt.figtext(0.03, 0.95, 'A', fontsize=16, fontweight='bold', color='black')

# 添加图例
plt.legend(title="Peptides", fontsize=12)

# 调整布局并保存图片
plt.tight_layout()
save_path = r"D:\xy\HeteroTCR-main\5folds_visual\picture\strict0_peptide_positive_prediction_confidence_curve_all.png"
plt.savefig(save_path, dpi=300)

# 显示图形
plt.show()

print(f"图像已保存到: {save_path}")
