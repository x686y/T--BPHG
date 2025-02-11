from PIL import Image
import matplotlib.pyplot as plt

# 定义图片路径
file_paths = [
    r'D:\xy\HeteroTCR-main\5folds_visual\picture\compareNetbalAUROC.png',
    r'D:\xy\HeteroTCR-main\5folds_visual\picture\compareNetbalAUPRC.png',
    r'D:\xy\HeteroTCR-main\5folds_visual\picture\youGrap_Netbalfold0_tSNE\RAKFKQLL.png',
    r'D:\xy\HeteroTCR-main\5folds_visual\picture\t-seneNoGNN_Netbalfold0\RAKFKQLL.png'
]

# 打开图片并获取尺寸
images = [Image.open(fp) for fp in file_paths]
widths, heights = zip(*(img.size for img in images))

# 设定目标统一尺寸，选择最大宽度和高度
target_width = max(widths)
target_height = max(heights)

# 调整每张图片的大小为目标尺寸
resized_images = [img.resize((target_width, target_height)) for img in images]

# 计算合成图片的宽度和高度
# 水平拼接：宽度为两张图片的宽度之和；竖直拼接：高度为两行最大高度之和
combined_width = target_width * 2
combined_height = target_height * 2

# 创建新的空白图片
combined_image = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))

# 粘贴第一行的两张图片
x_offset = 0
combined_image.paste(resized_images[0], (x_offset, 0))
x_offset += target_width
combined_image.paste(resized_images[1], (x_offset, 0))

# 粘贴第二行的两张图片
x_offset = 0
y_offset = target_height  # 第二行的起始y坐标
combined_image.paste(resized_images[2], (x_offset, y_offset))
x_offset += target_width
combined_image.paste(resized_images[3], (x_offset, y_offset))

# 保存合并后的图片，并设置分辨率为 300 dpi
output_path = r'D:\xy\HeteroTCR-main\5folds_visual\picture\combineBalencepep.png'
combined_image.save(output_path, format='PNG', dpi=(300, 300))  # 指定 300 dpi
print(f'合并后的图片已保存至: {output_path}')

# 显示合成图片（可视化）
combined_image.show()  # 打开系统图片查看器显示图片

# 使用 matplotlib 进行可视化
plt.figure(figsize=(12, 8))  # 适当调整图的尺寸
plt.imshow(combined_image)
plt.axis('off')  # 隐藏坐标轴
plt.show()
