import matplotlib.pyplot as plt
import numpy as np

# peptide名称
peptides = ['FEDLRLLSF(1)', 'FEDLRVLSF(1)', 'GILGFVFTL(6)', 'LLWNGPMAV(6)', 'VLFGLGFAI(6)',
            'CINGVCWTV(6)', 'IVTDFSVIK(6)', 'AVFDRKSDAK(6)', 'KLGGALQAK(6)', 'RPPIFIRRL(6)',
            'GPRLGVRAT(6)', 'NLVPMVATV(6)', 'RLRAEAQVK(6)', 'SLFNTVATLY(6)', 'GLCTLVAML(6)',
            'RAKFKQLL(6)', 'RLPGVLPRA(6)', 'CTELKLSDY(7)', 'YLQPRTFLL(7)', 'KSKRTPMGF(7)',
            'RFPLTFGWCF(7)', 'HPVTKYIM(7)', 'SPRWYFYYL(7)', 'ELAGIGILTV(7)', 'ATDALMTGF(7)',
            'DATYQRTRALVR(9)']

# AUPRC值
auprc_values = [0.856, 0.683, 0.655, 0.585, 0.521, 0.687, 0.695, 0.592, 0.621, 0.425,
                0.598, 0.553, 0.620, 0.466, 0.599, 0.752, 0.629, 0.567, 0.749, 0.564,
                0.491, 0.488, 0.611, 0.735, 0.608, 0.636]

# 统一颜色设置
bar_color = (30/255, 128/255, 184/255)

# 设置图形大小
plt.figure(figsize=(10, 8))

# 绘制条形图
bars = plt.barh(peptides, auprc_values, color=bar_color)

# 为每个条形图标注AUPRC值
for bar in bars:
    plt.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height() / 2,
             f'{bar.get_width():.3f}', va='center', ha='right', fontsize=14, color='white')

# 添加标题和标签
plt.title('AUPRC per peptide for peptides', fontsize=18, fontweight='bold')
plt.xlabel('AUPRC', fontsize=18, fontweight='bold')
plt.ylabel('Peptide(L_min)', fontsize=18, fontweight='bold')

# 在图的左上方添加字母"B"
# plt.text(-0.05, 1.05, 'D', fontsize=22, fontweight='bold', va='center', ha='center', transform=plt.gca().transAxes)

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
save_path = r'D:\xy\HeteroTCR-main\5folds_visual\picture\netstrict0LevepepAUPRC.png'
plt.savefig(save_path, dpi=300)

# 显示图形
plt.show()
