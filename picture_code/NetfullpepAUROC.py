import matplotlib.pyplot as plt
import numpy as np

# peptide名称
peptides = ['GILGFVFTL', 'RAKFKQLL', 'KLGGALQAK', 'AVFDRKSDAK', 'ELAGIGILTV',
            'NLVPMVATV', 'IVTDFSVIK', 'LLWNGPMAV', 'CINGVCWTV', 'GLCTLVAML',
            'SPRWYFYYL', 'ATDALMTGF', 'DATYQRTRALVR', 'KSKRTPMGF', 'YLQPRTFLL',
            'HPVTKYIM', 'RFPLTFGWCF', 'CTELKLSDY', 'GPRLGVRAT', 'RLRAEAQVK',
            'RLPGVLPRA', 'SLFNTVATLY', 'RPPIFIRRL', 'FEDLRLLSF', 'VLFGLGFAI',
            'FEDLRVLSF']

# AUROC值
auroc_values = [0.902, 0.916, 0.796, 0.743, 0.947, 0.775, 0.756, 0.889,
                0.908, 0.873, 0.866, 0.874, 0.832, 0.816, 0.903, 0.904,
                0.896, 0.825, 0.767, 0.842, 0.740, 0.669, 0.817, 0.851,
                0.892, 0.786]

# 统一颜色设置
bar_color = (30/255, 128/255, 184/255)

# 设置图形大小
plt.figure(figsize=(10,8))

# 绘制条形图
bars = plt.barh(peptides, auroc_values, color=bar_color)

# 为每个条形图标注AUROC值
for bar in bars:
    plt.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height() / 2,
             f'{bar.get_width():.3f}', va='center', ha='right', fontsize=14, color='white')

# 添加标题和标签
plt.title('NetTCR_full', fontsize=18, fontweight='bold')
plt.xlabel('AUROC', fontsize=18, fontweight='bold')
plt.ylabel('Peptide', fontsize=18, fontweight='bold')
# 在图的左上方添加字母"B"
# plt.text(-0.05, 1.05, 'C', fontsize=22, fontweight='bold', va='center', ha='center', transform=plt.gca().transAxes)

# 设置x轴和y轴刻度标签的字体大小
plt.tick_params(axis='x', labelsize=18)  # x轴刻度字体大小
plt.tick_params(axis='y', labelsize=18)  # y轴刻度字体大小
# 设置边框为黑色，并调整边框的粗细
ax = plt.gca()  # 获取当前坐标轴
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # 设置边框颜色为黑色
    spine.set_linewidth(1)  # 设置边框宽度

# 设置图的边界，避免图形被截断
plt.tight_layout()

# 保存图像到指定路径
save_path = r'D:\xy\HeteroTCR-main\5folds_visual\picture\netfullpepAUC.png'
plt.savefig(save_path, dpi=300)

# 显示图形
plt.show()
