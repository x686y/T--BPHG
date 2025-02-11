import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.patches import Patch

# 模拟每个模型在5折交叉验证中得到的ROC-AUC值
auc_baseline = [0.678, 0.886, 0.849, 0.920, 0.833]  # Baseline的AUC值
auc_taβ_bphg = [0.834, 0.902, 0.848, 0.924, 0.878]  # Tαβ-BPHG的AUC值
auc_no_gnn = [0.523, 0.529, 0.543, 0.638, 0.532]  # No_GNN的AUC值

# 将所有模型的AUC值放入一个列表
auc_values = [auc_baseline, auc_taβ_bphg, auc_no_gnn]
model_names = ['Baseline', 'Tαβ-BPHG', 'No_GNN']  # 模型名称

# 绘制箱线图
plt.figure(figsize=(3, 4))
boxplot = plt.boxplot(auc_values, labels=model_names, patch_artist=True, showfliers=False,
                      medianprops={'color': 'black', 'linewidth': 1.2})  # 中位线改为黑色

# 设置箱体颜色（自定义颜色）
colors = [(213/255, 105/255, 93/255), (245/255, 176/255, 65/255), (246/255, 218/255, 101/255)]  # 三个模型的新颜色
for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)  # 设置透明度

# 在每个箱线图上添加散布的彩色点，颜色设置为自定义的颜色，并增加透明度
for i, data in enumerate(auc_values):
    x = np.random.normal(i + 1, 0.04, size=len(data))  # 添加一些随机扰动以避免重叠
    plt.scatter(x, data, color=colors[i], edgecolor='black', alpha=0.8, s=40)  # 设置散点颜色和边框

# 设置图表标题和轴标签
plt.title('NetTCR_bal', fontsize=12)

plt.ylabel('AUC', fontsize=12)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)  # 调整Y轴刻度字体大小
# plt.text(-0.05, 1.05, 'B', fontsize=14, fontweight='bold', va='center', ha='center', transform=plt.gca().transAxes)

# 计算P值并显示
p_value_baseline_vs_taβ_bphg = stats.ttest_ind(auc_baseline, auc_taβ_bphg).pvalue
p_value_taβ_bphg_vs_no_gnn = stats.ttest_ind(auc_taβ_bphg, auc_no_gnn).pvalue
# 打印P值到控制台
print(f"P值 Baseline vs Tαβ-BPHG: {p_value_baseline_vs_taβ_bphg:.5f}")
print(f"P值 Tαβ-BPHG vs No_GNN: {p_value_taβ_bphg_vs_no_gnn:.5f}")
# # 显示P值
# plt.text(1.5, 0.85, f'p = {p_value_baseline_vs_taβ_bphg:.3f}', ha='center', va='bottom', fontsize=14)
# plt.text(2.5, 0.85, f'p = {p_value_taβ_bphg_vs_no_gnn:.3f}', ha='center', va='bottom', fontsize=14)

# 计算并打印每组数据的Q1、Q3和中位数
for i, data in enumerate(auc_values):
    q1 = np.percentile(data, 25)
    median = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    print(f"{model_names[i]}: Q1={q1:.4f}, Median={median:.4f}, Q3={q3:.4f}")

# 创建保存路径
save_dir = r'D:\xy\HeteroTCR-main\5folds_visual\picture'
os.makedirs(save_dir, exist_ok=True)  # 如果路径不存在，则创建该路径

# 保存图像
save_path = os.path.join(save_dir, 'Xiaorongnetbal_auc_boxplot.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')

# 自动调整布局
plt.tight_layout()  # 自动调整布局，防止标签被裁剪

# 显示图表
plt.grid(True)
plt.show()
print(f'图像已保存至: {save_path}')
