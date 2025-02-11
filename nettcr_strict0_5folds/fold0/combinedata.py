# import pandas as pd
#
# # 定义文件路径
# # 定义文件路径
# train_file = r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\fold0\train_data.tsv"
# test_file = r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\fold0\test_data.tsv"
# output_file = r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\fold0\netstrictfold0.tsv"
#
# # 读取文件并处理
# def process_file(filepath):
#     # 读取文件
#     data = pd.read_csv(filepath, sep="\t")
#     # 拼接 tra 和 trb 列，生成新的列 tcrab
#     data['tcrab'] = data['tra']  + data['trb']
#     # 删除原来的 tra 和 trb 列
#     data = data.drop(columns=['tra', 'trb'])
#     return data
#
# # 处理 train_file 和 test_file
# train_data = process_file(train_file)
# test_data = process_file(test_file)
#
# # 合并两个数据集
# combined_data = pd.concat([train_data, test_data], ignore_index=True)
#
# # 保存到新文件
# combined_data.to_csv(output_file, sep="\t", index=False)
#
# print(f"文件已成功合并并保存到：{output_file}")


import pandas as pd

# 定义文件路径
train_file = r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\fold0\train_data.tsv"
test_file = r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\fold0\test_data.tsv"
output_train_file = r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\fold0\train_data_processed.tsv"
output_test_file = r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\fold0\test_data_processed.tsv"

# 处理文件
def process_and_save_file(input_filepath, output_filepath):
    # 读取文件
    data = pd.read_csv(input_filepath, sep="\t")
    # 拼接 tra 和 trb 列，生成新的列 tcrab
    data['tcrab'] = data['tra'] + data['trb']
    # 删除原来的 tra 和 trb 列
    data = data.drop(columns=['tra', 'trb'])
    # 保存到新的文件
    data.to_csv(output_filepath, sep="\t", index=False)
    print(f"文件已处理并保存到：{output_filepath}")

# 处理 train_file 和 test_file
process_and_save_file(train_file, output_train_file)
process_and_save_file(test_file, output_test_file)