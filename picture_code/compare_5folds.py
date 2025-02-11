import matplotlib.pyplot as plt
import numpy as np

# 模型名称和数据
models = ['NetTCR2.0', 'K-NN', 'XGBoost', 'Logistic Regression', 'HGTCR']
auc_values = [0.889, 0.802, 0.896, 0.837, 0.926]

# 设置柱状图的宽度和位置
x = np.arange(len(models)) * 0.5  # 缩小条形图间距
width = 0.3  # 保持条形图宽度

# 每个模型不同的颜色 (替换为新颜色代码)
colors = [
    [0.6588, 0.7765, 0.8627],  # R168 G198 B220
    [0.9569, 0.6745, 0.6745],  # R244 G172 B172
    [1.0000, 0.7961, 0.6667],  # R255 G203 B170
    [0.8471, 0.7569, 0.9098],  # R216 G193 B232
    [0.6980, 0.8510, 0.8510]   # R178 G217 B217
]

# 创建图表
fig, ax = plt.subplots(figsize=(8, 6))  # 调整图表宽度，使其适应更紧凑的条形图

# 绘制条形图，不带误差线
bars_auc = ax.bar(x, auc_values, width, color=colors)

# 添加黄色折线图
ax.plot(x, auc_values, color='orange', marker='o', markersize=8, label='Trend Line (AUC)')

# 在每个黄色圆点上标注 AUC 值
for i, value in enumerate(auc_values):
    ax.text(x[i], value + 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=12, color='orange')

# 为每个模型添加不同颜色的图例
for i, model in enumerate(models):
    ax.bar(0, 0, color=colors[i], label=model)  # 创建伪条形图用于图例

# 设置坐标轴标签和标题
ax.set_ylabel('AUC', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=14)

# 设置 x 轴范围，使得条形图之间的间隔一致
ax.set_xlim(-0.25, x[-1] + 0.25)

# 设置 y 轴刻度字体大小和加粗
ax.tick_params(axis='y', labelsize=14, width=2)

# 设置 y 轴范围
ax.set_ylim(0, 1.1)

# 增加图例，移动到右上角，避免遮挡柱子
ax.legend(title="Models", fontsize=12, title_fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))

# 保存图表
save_path = r'D:\xy\HeteroTCR-main\5folds_visual\picture\compare5folds.png'
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f'图像已保存至: {save_path}')
