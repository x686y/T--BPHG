import pandas as pd
import matplotlib.pyplot as plt

# 样本数据
data = {
    'Sample': ['GILGFVFTL', 'LLWNGPMAV', 'IVTDFSVIK', 'ELAGIGILTV'],
    'TCRα': [0.453, 0.527, 0.507, 0.453],  # 对第三和第四个样本的weights_tra做转换
    'TCRβ': [0.547, 0.473, 0.493, 0.547]   # 对第三和第四个样本的weights_trb做转换
}

# 将数据加载到 DataFrame 中
df = pd.DataFrame(data)

# 设置图的大小
plt.figure(figsize=(10, 8))

# 设置条形图
bar_width = 0.25  # 将条形图宽度调小
x = [i * 0.8 for i in range(len(df))]  # 缩小 x 的步长，减少间距

# 颜色定义（使用图中画圈的颜色）
color_tra = (170/255, 220/255, 224/255)  # R90 G148 B185
color_trb = (114/255, 188/255, 213/255)  # R42 G99 B152

# 绘制 weights_tra 和 weights_trb 的条形图（去掉边框线）
plt.bar(x, df['TCRα'], width=bar_width, label='TCRα', color=color_tra)
plt.bar([i + bar_width for i in x], df['TCRβ'], width=bar_width, label='TCRβ', color=color_trb)

# 在条形图上添加值标签，确保数字显示在条形图正上方
for i, (tra, trb) in enumerate(zip(df['TCRα'], df['TCRβ'])):
    plt.text(x[i], tra + 0.005, f'{tra:.3f}', ha='center', va='bottom', color='black', fontsize=14)  # TCRα标签
    plt.text(x[i] + bar_width, trb + 0.005, f'{trb:.3f}', ha='center', va='bottom', color='black', fontsize=14)  # TCRβ标签

# 设置图例，放到图片外面右上角并向右移动
plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)
# 设置标签和标题
plt.ylabel('Average weight value', fontsize=16, fontweight='bold')  # y轴标签字体大小
plt.xticks([i + bar_width / 2 for i in x], df['Sample'], fontsize=14)  # x轴刻度字体大小
plt.yticks(fontsize=14)  # y轴刻度字体大小

# 保存图片
output_path = r'D:\xy\HeteroTCR-main\5folds_visual\picture\Weight Value.png'
plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')  # bbox_inches='tight'确保图例完全显示
print(f'图片已保存至: {output_path}')

# 显示图形
plt.show()
