from PIL import Image
import matplotlib.pyplot as plt

# 定义图片路径，只保留前四张图片
file_paths = [
    r'D:\xy\HeteroTCR-main\5folds_visual\picture\compareNetfullAUROC.png',
    r'D:\xy\HeteroTCR-main\5folds_visual\picture\compareNetfullAUPRC.png',
    r'D:\xy\HeteroTCR-main\5folds_visual\picture\netfullpepAUC.png',
    r'D:\xy\HeteroTCR-main\5folds_visual\picture\netfullpepAUPRC.png'
]

# 打开图片并获取尺寸
images = [Image.open(fp) for fp in file_paths]
widths, heights = zip(*(img.size for img in images))

# 计算合成图片的宽度和高度
# 第一行：前两个图片拼接
row1_width = sum(widths[:2])  # 前两张图片的宽度相加
row1_height = max(heights[:2])  # 取前两张图片的最大高度

# 第二行：后两个图片拼接
row2_width = sum(widths[2:])  # 后两张图片的宽度相加
row2_height = max(heights[2:])  # 取后两张图片的最大高度

# 合成图片的总宽度和总高度
total_width = max(row1_width, row2_width)  # 取最大宽度
total_height = row1_height + row2_height  # 高度是两行的总和

# 创建新的空白图片
combined_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

# 粘贴前两张图片（第一行）
x_offset = 0
for img in images[:2]:
    combined_image.paste(img, (x_offset, 0))
    x_offset += img.width  # 更新x偏移量，确保图片在一行中依次排列

# 粘贴后两张图片（第二行）
y_offset = row1_height  # 第二行的y偏移量等于第一行的高度
x_offset = 0  # 第二行从x=0开始
for idx, img in enumerate(images[2:]):
    if idx == 1:  # 如果是第四张图片（第二行的第二张），调整水平偏移量
        x_offset += 140  # 增加50像素的水平偏移量（可以根据需要调整数值）
    combined_image.paste(img, (x_offset, y_offset))
    x_offset += img.width  # 更新x偏移量，确保图片在第二行中依次排列

# 保存合并后的图片，并设置分辨率为 300 dpi
output_path = r'D:\xy\HeteroTCR-main\5folds_visual\picture\combineSeenpep.png'
combined_image.save(output_path, format='PNG', dpi=(300, 300))  # 指定 300 dpi
print(f'合并后的图片已保存至: {output_path}')

# 显示合成图片（可视化）
combined_image.show()  # 打开系统图片查看器显示图片

# 使用 matplotlib 进行可视化
plt.figure(figsize=(16, 12))
plt.imshow(combined_image)
plt.axis('off')  # 隐藏坐标轴
plt.show()
