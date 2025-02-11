from PIL import Image
import matplotlib.pyplot as plt

# 定义图片路径
file_paths = [
    r'D:\xy\HeteroTCR-main\5folds_visual\picture\Netfulldata.png',
    r'D:\xy\HeteroTCR-main\5folds_visual\picture\Netbaldata.png',
    r'D:\xy\HeteroTCR-main\5folds_visual\picture\Netstrictdata.png'
]

# 打开图片并获取尺寸
images = [Image.open(fp) for fp in file_paths]
widths, heights = zip(*(img.size for img in images))

# 计算合成图片的宽度和高度
total_width = max(widths)  # 取所有图片的最大宽度
total_height = sum(heights)  # 总高度为所有图片高度之和

# 创建新的空白图片
combined_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

# 粘贴图片（从上到下）
y_offset = 0
for img in images:
    combined_image.paste(img, (0, y_offset))
    y_offset += img.height  # 更新y偏移量，确保图片按顺序从上到下排列

# 保存合并后的图片，并设置分辨率为 300 dpi
output_path = r'D:\xy\HeteroTCR-main\5folds_visual\picture\combinethreeData.png'
combined_image.save(output_path, format='PNG', dpi=(300, 300))  # 指定 300 dpi
print(f'合并后的图片已保存至: {output_path}')

# 显示合成图片（可视化）
combined_image.show()  # 打开系统图片查看器显示图片

# 使用 matplotlib 进行可视化
plt.figure(figsize=(8, 12))  # 调整显示比例
plt.imshow(combined_image)
plt.axis('off')  # 隐藏坐标轴
plt.show()
