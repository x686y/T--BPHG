import pandas as pd
import matplotlib.pyplot as plt

# 文件路径
train_file = r'D:\xy\HeteroTCR-main\data\nettcrfull5folds\fold0\train_data.tsv'
test_file = r'D:\xy\HeteroTCR-main\data\nettcrfull5folds\fold0\test_data.tsv'
output_path = r'D:\xy\HeteroTCR-main\5folds_visual\picture\Netfulldata.png'

# 读取数据
train_data = pd.read_csv(train_file, sep='\t')
test_data = pd.read_csv(test_file, sep='\t')

# 合并两个文件
combined_data = pd.concat([train_data, test_data], ignore_index=True)

# 确保包含 'peptide' 和 'Binding' 列
if 'peptide' not in combined_data.columns or 'Binding' not in combined_data.columns:
    raise ValueError("Missing 'peptide' or 'Binding' column in the input files.")

# 按肽（peptide）和绑定状态（Binding）统计数量
peptide_counts = combined_data.groupby(['peptide', 'Binding']).size().unstack(fill_value=0)

# 重命名列方便理解
peptide_counts.columns = ['Negative', 'Positive']

# 按阳性样本数量降序排列
peptide_counts = peptide_counts.sort_values('Positive', ascending=False)

# 绘制条形图
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

# 绘制正样本条形图（颜色改为指定RGB颜色）
ax.bar(
    peptide_counts.index,
    peptide_counts['Positive'],
    color=(244/255, 109/255, 67/255),
    label='Positive Binding TCRs',
    alpha=0.7
)

# 绘制负样本条形图（颜色改为指定RGB颜色）
ax.bar(
    peptide_counts.index,
    peptide_counts['Negative'],
    bottom=peptide_counts['Positive'],
    color=(254/255, 224/255, 139/255),
    label='Negative Binding TCRs',
    alpha=0.7
)
# 缩短横坐标空白区域
plt.margins(x=0.02)  # 设置两侧留出 2% 的边距

# 设置轴标签和标题
ax.set_ylabel("Number of Samples", fontsize=14,fontweight='bold')
# ax.set_xlabel("Peptides", fontsize=14,fontweight='bold')
plt.title("NetTCR_full", fontsize=14,fontweight='bold')
# 在图的左上方添加字母"B"
# plt.text(-0.02, 1.05, 'A', fontsize=20, fontweight='bold', va='center', ha='center', transform=plt.gca().transAxes)

# 调整X轴标签（倾斜45度）
ax.set_xticks(range(len(peptide_counts.index)))
ax.set_xticklabels(peptide_counts.index, rotation=45, fontsize=14)

legend = plt.legend(
    title="Binding Type",
    fontsize=14,  # 设置图例标签字体大小
    title_fontsize=14  # 设置图例标题字体大小
)


# 调整布局和保存图片
plt.tight_layout()
plt.savefig(output_path, dpi=300)

# 显示图像
# plt.show()

print(f"图像已保存到: {output_path}")
