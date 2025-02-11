from PIL import Image
import matplotlib.pyplot as plt

# 定义图片路径，只保留前两张图片
file_paths = [
    r'D:\xy\HeteroTCR-main\5folds_visual\picture\netfullpepAUC.png',
    r'D:\xy\HeteroTCR-main\5folds_visual\picture\netfullpepAUPRC.png'
]

# 打开图片并获取尺寸
images = [Image.open(fp) for fp in file_paths]
widths, heights = zip(*(img.size for img in images))

# 计算合成图片的宽度和高度
# 第一行：前两个图片拼接
row1_width = sum(widths)  # 只有两张图片，宽度相加即可
row1_height = max(heights)  # 取最大高度作为行高

# 创建新的空白图片
combined_image = Image.new('RGB', (row1_width, row1_height), (255, 255, 255))

# 粘贴前两张图片
x_offset = 0
for img in images:
    combined_image.paste(img, (x_offset, 0))
    x_offset += img.width  # 更新x偏移量，确保图片在一行中依次排列

# 保存合并后的图片，并设置分辨率为 300 dpi
output_path = r'D:\xy\HeteroTCR-main\5folds_visual\picture\combineNetfullpep.png'
combined_image.save(output_path, format='PNG', dpi=(300, 300))  # 指定 300 dpi
print(f'合并后的图片已保存至: {output_path}')

# 显示合成图片（可视化）
combined_image.show()  # 打开系统图片查看器显示图片

# 使用 matplotlib 进行可视化
plt.figure(figsize=(10, 8))
plt.imshow(combined_image)
plt.axis('off')  # 隐藏坐标轴
plt.show()
