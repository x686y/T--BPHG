from PIL import Image
import matplotlib.pyplot as plt

# 定义图片路径
file_paths = [
    r'D:\xy\HeteroTCR-main\5folds_visual\picture\compareNetStrictAUROC.png',
    r'D:\xy\HeteroTCR-main\5folds_visual\picture\compareNetStrictAUPRC.png',
    r'D:\xy\HeteroTCR-main\5folds_visual\picture\netstrict0LevepepAUC.png',
    r'D:\xy\HeteroTCR-main\5folds_visual\picture\netstrict0LevepepAUPRC.png',
]

# 打开图片并获取尺寸
images = [Image.open(fp) for fp in file_paths]
widths, heights = zip(*(img.size for img in images))

# 计算合成图片的宽度和高度
# 第一行宽度：前两张图片的宽度之和，最大高度：前两张图片的最大高度
first_row_width = widths[0] + widths[1]
first_row_height = max(heights[0], heights[1])

# 第二行宽度：后两张图片的宽度之和，最大高度：后两张图片的最大高度
second_row_width = widths[2] + widths[3]
second_row_height = max(heights[2], heights[3])

# 合并图片的总宽度和总高度
combined_width = max(first_row_width, second_row_width)
combined_height = first_row_height + second_row_height

# 创建新的空白图片
combined_image = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))

# 粘贴第一行的图片（前两个）
x_offset = 0
combined_image.paste(images[0], (x_offset, 0))  # 粘贴第一张图片
x_offset += images[0].width
combined_image.paste(images[1], (x_offset, 0))  # 粘贴第二张图片

# 粘贴第二行的图片（后两个）
x_offset = 0
combined_image.paste(images[2], (x_offset, first_row_height))  # 粘贴第三张图片（位置在第一行图片的下方）
x_offset += images[2].width + 50  # 添加额外的水平偏移量
combined_image.paste(images[3], (x_offset, first_row_height))  # 粘贴第四张图片（位置在第一行图片的下方）

# 保存合并后的图片，并设置分辨率为 300 dpi
output_path = r'D:\xy\HeteroTCR-main\5folds_visual\picture\combineUnseenpep.png'
combined_image.save(output_path, format='PNG', dpi=(300, 300))  # 指定 300 dpi
print(f'合并后的图片已保存至: {output_path}')

# 显示合成图片（可视化）
combined_image.show()  # 打开系统图片查看器显示图片

# 使用 matplotlib 进行可视化
plt.figure(figsize=(14, 10))  # 适当调整图的尺寸
plt.imshow(combined_image)
plt.axis('off')  # 隐藏坐标轴
plt.show()
