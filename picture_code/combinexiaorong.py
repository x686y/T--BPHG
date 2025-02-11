from PIL import Image
import matplotlib.pyplot as plt

# 定义图片路径
file_paths = [
    r'D:\xy\HeteroTCR-main\5folds_visual\picture\Xiaorongnetfull_auc_boxplot.png',
    r'D:\xy\HeteroTCR-main\5folds_visual\picture\Xiaorongnetbal_auc_boxplot.png',
    r'D:\xy\HeteroTCR-main\5folds_visual\picture\XiaorongnetStrict_auc_boxplot.png',
    r'D:\xy\HeteroTCR-main\5folds_visual\picture\Xiaorongnetfull_auprc_boxplot.png',
    r'D:\xy\HeteroTCR-main\5folds_visual\picture\Xiaorongnetbal_auprc_boxplot.png',
    r'D:\xy\HeteroTCR-main\5folds_visual\picture\XiaorongnetStrict_auprc_boxplot.png'
]

# 打开图片并获取尺寸
images = [Image.open(fp) for fp in file_paths]
widths, heights = zip(*(img.size for img in images))

# 计算合成图片的宽度和高度
# 第一行：前三个图片拼接
row1_width = sum(widths[:3])
# 第二行：后三个图片拼接
row2_width = sum(widths[3:])

# 取每行的最大高度
row1_height = max(heights[:3])
row2_height = max(heights[3:])

# 总高度为两行的高度之和
total_height = row1_height + row2_height

# 创建新的空白图片
combined_image = Image.new('RGB', (max(row1_width, row2_width), total_height), (255, 255, 255))

# 粘贴第一行的三个图片
x_offset = 0
for img in images[:3]:
    combined_image.paste(img, (x_offset, 0))
    x_offset += img.width

# 粘贴第二行的三个图片
x_offset = 0  # 第二行从最左边开始
y_offset = row1_height
for img in images[3:]:
    combined_image.paste(img, (x_offset, y_offset))
    x_offset += img.width  # 更新x偏移

# 保存合并后的图片，并设置分辨率为 300 dpi
output_path = r'D:\xy\HeteroTCR-main\5folds_visual\picture\combinexiaorong.png'
combined_image.save(output_path, format='PNG', dpi=(300, 300))  # 指定 300 dpi
print(f'合并后的图片已保存至: {output_path}')

# 显示合成图片（可视化）
combined_image.show()  # 打开系统图片查看器显示图片

# 使用 matplotlib 进行可视化
plt.figure(figsize=(12, 10))
plt.imshow(combined_image)
plt.axis('off')  # 隐藏坐标轴
plt.show()
