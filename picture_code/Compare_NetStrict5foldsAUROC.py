import matplotlib.pyplot as plt
import numpy as np

# 模型名称
models = ['TITAN', 'epiTCR', 'TEINet', 'MixTCRpred', 'NetTCR-2.2',
          'TSpred_CNN', 'TSpred_att', 'TSpred_ens', 'HeteroTCR', 'Tαβ-BPHG']

# AUROC值和标准差
auroc_values = [0.518, 0.584, 0.646, 0.601, 0.608, 0.628, 0.645, 0.651, 0.622, 0.671]
std_devs = [0.031, 0.023, 0.021, 0.048, 0.038, 0.041, 0.039, 0.041, 0.025, 0.034]

# 10种颜色，使用RGB格式
colors = [(254/255,254/255,215/255),  # #1f77b4 蓝色
          (238/255,248/255,180/255),  # #ff7f0e 橙色
          (205/255,235/255,179/255),  # #2ca02c 绿色
          (149/255,213/255,184/255),  # #d62728 红色
          (91/255,191/255,192/255),  # #9467bd 紫色
          (48/255,165/255,194/255),  # #8c564b 棕色
          (30/255,128/255,184/255),  # #e377c2 粉色
          (34/255,84/255,163/255),  # #7f7f7f 灰色
          (33/255,49/255,140/255),  # #bcbd22 黄绿色
          (8/255,29/255,89/255)]  # #17becf 天蓝色

# 设置图形大小
plt.figure(figsize=(12, 8))

# 绘制条形图，不添加误差线，使用指定的颜色
bars = plt.bar(models, auroc_values, color=colors)

# 添加标题和标签
plt.title('NetTCR_strict',fontsize=18 ,fontweight='bold')
plt.xlabel('Model',fontsize=18 ,fontweight='bold')
plt.ylabel('AUROC',fontsize=18 ,fontweight='bold')

# 在图的左上方添加字母"A"
# plt.text(-0.05, 1.05, 'A', fontsize=22, fontweight='bold', va='center', ha='center', transform=plt.gca().transAxes)

# 添加每个模型的图例信息
legend_labels = [f'{models[i]} (AUC={auroc_values[i]:.3f} ± {std_devs[i]:.3f})' for i in range(len(models))]

# 创建图例并将其放置在指定位置
plt.legend(bars, legend_labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)
# 设置边框为黑色，并调整边框的粗细
ax = plt.gca()  # 获取当前坐标轴
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # 设置边框颜色为黑色
    spine.set_linewidth(1)  # 设置边框宽度

# 设置x轴标签旋转角度，防止重叠
plt.xticks(rotation=45, ha='right')

# 设置x轴和y轴刻度标签的字体大小
plt.tick_params(axis='x', labelsize=18)  # x轴刻度字体大小
plt.tick_params(axis='y', labelsize=18)  # y轴刻度字体大小

# 设置图的边界，避免图形被截断
plt.tight_layout()

# 保存图像到指定路径
save_path = r'D:\xy\HeteroTCR-main\5folds_visual\picture\compareNetStrictAUROC.png'
plt.savefig(save_path, dpi=300)

# 显示图形
plt.show()
