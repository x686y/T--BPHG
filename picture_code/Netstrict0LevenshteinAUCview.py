import matplotlib.pyplot as plt
import numpy as np

# peptide名称
peptides = ['FEDLRVLSF(1)', 'FEDLRLLSF(1)', 'GLCTLVAML(6)', 'RAKFKQLL(6)', 'RLPGVLPRA(6)',
            'GPRLGVRAT(6)', 'NLVPMVATV(6)', 'RLRAEAQVK(6)', 'SLFNTVATLY(6)', 'AVFDRKSDAK(6)',
            'KLGGALQAK(6)', 'RPPIFIRRL(6)', 'CINGVCWTV(6)', 'IVTDFSVIK(6)', 'GILGFVFTL(6)',
            'LLWNGPMAV(6)', 'VLFGLGFAI(6)', 'CTELKLSDY(7)', 'YLQPRTFLL(7)', 'KSKRTPMGF(7)',
            'RFPLTFGWCF(7)', 'HPVTKYIM(7)', 'SPRWYFYYL(7)', 'ELAGIGILTV(7)', 'ATDALMTGF(7)',
            'DATYQRTRALVR(9)']

# AUROC值
auroc_values = [0.658, 0.853, 0.638, 0.771, 0.622, 0.570, 0.572, 0.622, 0.440,
                0.629, 0.652, 0.360, 0.705, 0.732, 0.700, 0.634, 0.398, 0.586,
                0.727, 0.605, 0.472, 0.475, 0.626, 0.786, 0.614, 0.652]

# 统一颜色设置
bar_color = (30/255, 128/255, 184/255)

# 设置图形大小
plt.figure(figsize=(10, 8))

# 绘制条形图
bars = plt.barh(peptides, auroc_values, color=bar_color)

# 为每个条形图标注AUROC值
for bar in bars:
    plt.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height() / 2,
             f'{bar.get_width():.3f}', va='center', ha='right', fontsize=14, color='white')

# 添加标题和标签
plt.title('AUROC per peptide for peptides', fontsize=18, fontweight='bold')
plt.xlabel('AUROC', fontsize=18, fontweight='bold')
plt.ylabel('Peptide(L_min)', fontsize=18, fontweight='bold')

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
save_path = r'D:\xy\HeteroTCR-main\5folds_visual\picture\netstrict0LevepepAUC.png'
plt.savefig(save_path, dpi=300)

# 显示图形
plt.show()
