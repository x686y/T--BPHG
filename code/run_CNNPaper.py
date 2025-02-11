

import os, sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras_metrics as km
# import keras.metrics as km
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, concatenate, Flatten, Embedding, Dropout
from keras.optimizers import adam_v2
from keras.initializers import glorot_normal
from keras.activations import sigmoid
from keras.callbacks import EarlyStopping
import random
import pickle
from configPaper import *

def seed_tensorflow(seed=42):
    random.seed(seed)  # 设置 Python 标准库中的 random 模块的随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置 Python 的 hash 函数的随机种子，这有助于确保字典等基于哈希的数据结构的行为是可重复的
    np.random.seed(seed)  # 设置 NumPy 库的随机种子
    tf.random.set_seed(seed)  # 设置 TensorFlow 的随机种子


seed_tensorflow()  # 调用函数，设置随机种子为 42（这是一个常用的默认值）

# config配置项目中涉及的各种路径和参数
# 参数转换
epochs = int(args.epochs)  # 从某个参数集合（args）中获取名为epochs的参数，并将其转换为整数
batchsize = int(args.batchsize)
lr = float(args.cnnlearningrate)

# 路径构建
root = os.path.join(args.pridir, args.secdir, args.terdir)  # ：使用os.path.join函数将args中的三个参数组合成一个完整的路径，代表了数据或模型存储的根目录。
train_csv = os.path.join(root, 'train_data.tsv')  # 分别根据root路径和文件名'train.tsv'、'test.tsv'生成训练集和测试集的CSV文件路径。
test_csv = os.path.join(root, 'test_data.tsv')

root_model = args.modeldir  # args中获取模型存储的根目录路径
model_dir = str(args.secdir) + '_' + str(
    args.terdir) + '_PaperCNN'  # 根据args中的secdir和terdir参数，以及字符串'_CNN'，生成一个用于标识模型存储位置的目录名。这个目录名将用于在root_model指定的根目录下创建一个子目录来保存模型。
model_dir2 = str(args.secdir) + '_' + str(args.terdir)
save_model_path = os.path.join(root_model, model_dir)  # 将root_model和model_dir组合成一个完整的路径，将用于保存训练好的模型。
root_history = args.hisdir  # 从args中获取用于保存训练历史记录（如损失和准确率）的根目录路径

# 文件路径设置
train_path_pickle_cdr3 = os.path.join(root,'PaperCNN_feature_cdr3_train.pickle')  # 分别设置了训练数据中CDR3特征和肽段特征的 pickle 文件路径。这些文件可能包含了通过某种预处理（如使用卷积神经网络CNN提取的特征）得到的特征数据，以 pickle 格式存储。pickle 是 Python 的一种序列化对象结构的方式，可以方便地将对象保存到文件中，并在需要时重新加载。
train_path_pickle_peptide = os.path.join(root, 'PaperCNN_feature_peptide_train.pickle')
test_path_pickle_cdr3 = os.path.join(root, 'PaperCNN_feature_cdr3_test.pickle')
test_path_pickle_peptide = os.path.join(root, 'PaperCNN_feature_peptide_test.pickle')

# Load data加载数据
print('Loading the train data..')
train_data = pd.read_csv(train_csv,
                         delimiter='\t')  # 从 CSV 文件中加载训练数据和测试数据，train_csv 和 test_csv 变量分别存储了训练集和测试集的 CSV 文件路径，这些路径是在之前的代码中通过组合不同的目录和文件名得到的。delimiter='\t' 参数指定了 CSV 文件中字段之间的分隔符是制表符（\t），加载的数据分别存储在 train_data 和 test_data 变量中，这两个变量现在包含了 DataFrame 对象，DataFrame 是 pandas 库中用于存储和操作结构化数据的主要数据结构。
print(f"Length of train_data: {len(train_data)}")  # 打印训练数据的长度

print('Loading the test data..')
test_data = pd.read_csv(test_csv, delimiter='\t')
print(f"Length of test_data: {len(test_data)}")  # 打印测试数据的长度
print(train_data.columns)
print(train_data.info())

def enc_list_bl_max_len(aa_seqs, blosum,
                        max_seq_len):  # 对一组氨基酸序列（aa_seqs）进行BLOSUM编码，并将编码后的序列填充到统一的最大长度（max_seq_len），blosum: 一个字典，键是氨基酸（AA）的缩写，值是该氨基酸对应的BLOSUM编码。BLOSUM编码是一种基于氨基酸替换频率的矩阵，常用于蛋白质序列的比对和分类。
    """
    @description:
                blosum encoding of a list of amino acid sequences with padding to a max length
    ----------
    @param:
                aa_seqs: list with AA sequences
                blosum: dictionnary: key=AA, value=blosum encoding
                max_seq_len: common length for padding
    ----------
    @Returns:
                enc_aa_seq : list of np.ndarrays containing padded, encoded amino acid sequences
    ----------
    """
    # encode sequences
    sequences = []  # 初始化一个空列表，用于存储每个氨基酸序列转换后的编码
    for seq in aa_seqs:  # 遍历给定的氨基酸序列列表
        e_seq = np.zeros((len(seq), len(
            blosum["A"])))  # 对于每个序列，初始化一个全零矩阵e_seq，其行数为序列长度，列数为BLOSUM矩阵中任一氨基酸向量的长度（这里以"A"为例，假设所有氨基酸的向量长度相同）
        count = 0  # 初始化计数器，用于跟踪当前处理的氨基酸在序列中的位置
        for aa in seq:  # 遍历当前序列中的每个氨基酸
            if aa in blosum:  # 检查当前氨基酸是否在BLOSUM字典（或类似的数据结构）中
                e_seq[count] = blosum[aa]  # 如果在，则将BLOSUM中对应的向量赋值给e_seq的当前行
                count += 1  # 计数器增加，以处理下一个氨基酸
            else:  # 如果当前氨基酸不在BLOSUM中
                print(seq)  # 打印出有问题的序列
                sys.stderr.write(
                    "Unknown amino acid in peptides: " + aa + ", encoding aborted!\n")  # 向标准错误输出错误信息，并指出哪个氨基酸未知，然后终止程序
                sys.exit(2)  # 退出程序，并返回状态码2，表示由于未知氨基酸而终止
        sequences.append(e_seq)  # 将处理好的编码矩阵e_seq添加到sequences列表中

    # pad sequences填充序列
    n_seqs = len(aa_seqs)  # 计算序列总数，用于后续的数组初始化
    n_features = sequences[0].shape[1]  # 提取一个序列编码矩阵的列数（即特征数），假设所有序列的编码长度相同
    enc_aa_seq = np.zeros((n_seqs, max_seq_len, n_features))  # 初始化一个全零的三维数组，用于存储所有序列的编码，包括必要的填充
    for i in range(0, n_seqs):  # 遍历每个序列的编码
        enc_aa_seq[i, :sequences[i].shape[0], :n_features] = sequences[
            i]  # 将每个序列的编码复制到enc_aa_seq的相应位置，对于较短的序列，其余部分保持为零（即填充）
    return enc_aa_seq  # 返回填充后的编码数组，数组包含所有序列的编码，每个序列都被填充到了统一的最大长度max_seq_len，并且每个氨基酸都被转换为了对应的BLOSUM编码向量


# 一个预定义的BLOSUM50矩阵，用于将氨基酸转换为特征向量，blosum50_20aa 是一个字典，它包含了20种标准氨基酸在BLOSUM50替换矩阵中的得分向量。每一行代码都定义了一个键（氨基酸的单字母缩写）和一个值（一个NumPy数组，表示该氨基酸在BLOSUM50矩阵中的得分向量）。这个得分向量是一个长度为20的数组，因为BLOSUM矩阵是一个20x20的矩阵，每种氨基酸都对应一行和一列。数组中的每个元素都表示当前氨基酸（键）与其他氨基酸（按照字典顺序排列）之间的替换得分。
blosum50_20aa = {
    'A': np.array((5, -2, -1, -2, -1, -1, -1, 0, -2, -1, -2, -1, -1, -3, -1, 1, 0, -3, -2, 0)),
    # A与A的替换得分为5，A与R的替换得分为-2，依此类推
    'R': np.array((-2, 7, -1, -2, -4, 1, 0, -3, 0, -4, -3, 3, -2, -3, -3, -1, -1, -3, -1, -3)),
    'N': np.array((-1, -1, 7, 2, -2, 0, 0, 0, 1, -3, -4, 0, -2, -4, -2, 1, 0, -4, -2, -3)),
    'D': np.array((-2, -2, 2, 8, -4, 0, 2, -1, -1, -4, -4, -1, -4, -5, -1, 0, -1, -5, -3, -4)),
    'C': np.array((-1, -4, -2, -4, 13, -3, -3, -3, -3, -2, -2, -3, -2, -2, -4, -1, -1, -5, -3, -1)),
    'Q': np.array((-1, 1, 0, 0, -3, 7, 2, -2, 1, -3, -2, 2, 0, -4, -1, 0, -1, -1, -1, -3)),
    'E': np.array((-1, 0, 0, 2, -3, 2, 6, -3, 0, -4, -3, 1, -2, -3, -1, -1, -1, -3, -2, -3)),
    'G': np.array((0, -3, 0, -1, -3, -2, -3, 8, -2, -4, -4, -2, -3, -4, -2, 0, -2, -3, -3, -4)),
    'H': np.array((-2, 0, 1, -1, -3, 1, 0, -2, 10, -4, -3, 0, -1, -1, -2, -1, -2, -3, 2, -4)),
    'I': np.array((-1, -4, -3, -4, -2, -3, -4, -4, -4, 5, 2, -3, 2, 0, -3, -3, -1, -3, -1, 4)),
    'L': np.array((-2, -3, -4, -4, -2, -2, -3, -4, -3, 2, 5, -3, 3, 1, -4, -3, -1, -2, -1, 1)),
    'K': np.array((-1, 3, 0, -1, -3, 2, 1, -2, 0, -3, -3, 6, -2, -4, -1, 0, -1, -3, -2, -3)),
    'M': np.array((-1, -2, -2, -4, -2, 0, -2, -3, -1, 2, 3, -2, 7, 0, -3, -2, -1, -1, 0, 1)),
    'F': np.array((-3, -3, -4, -5, -2, -4, -3, -4, -1, 0, 1, -4, 0, 8, -4, -3, -2, 1, 4, -1)),
    'P': np.array((-1, -3, -2, -1, -4, -1, -1, -2, -2, -3, -4, -1, -3, -4, 10, -1, -1, -4, -3, -3)),
    'S': np.array((1, -1, 1, 0, -1, 0, -1, 0, -1, -3, -3, 0, -2, -3, -1, 5, 2, -4, -2, -2)),
    'T': np.array((0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 2, 5, -3, -2, 0)),
    'W': np.array((-3, -3, -4, -5, -5, -1, -3, -3, -3, -3, -2, -3, -1, 1, -4, -4, -3, 15, 2, -3)),
    'Y': np.array((-2, -1, -2, -3, -3, -1, -2, -3, 2, -1, -1, -2, 0, 4, -3, -2, -2, 2, 8, -1)),
    'V': np.array((0, -3, -3, -4, -1, -3, -3, -4, -4, 4, 1, -3, 1, -1, -3, -2, 0, -3, -1, 5))
}

# Feature encoding对训练和测试数据集中的TCRB CDR3序列和肽序列进行特征编码，并将它们以及对应的结合信息（Binding）准备为机器学习模型可以处理的格式
print('Encoding the train data..')
tcrB_train = enc_list_bl_max_len(train_data.B3, blosum50_20aa, 23)

# print('-----------------------------')
# print(train_data.trb_cdr3)
# print('-----------------------------')
# print(train_data.tra_cdr3)
# print('-----------------------------')
# print('-----------------------------')
# print(tcrb_train.shape)
# print('-----------------------------')
# print(tcra_train.shape)
# print('-----------------------------')
# print(tcr_train.shape)
# print('-----------------------------')
pep_train = enc_list_bl_max_len(train_data.peptide, blosum50_20aa,15)  # 类似地，对train_data.peptide中的肽序列进行编码，最大长度为15
y_train = np.array(train_data.Binding)  # 将train_data.Binding（可能是一个列表或数组，包含了与TCRB CDR3和肽序列对应的结合信息）转换为NumPy数组

print('Encoding the test data..')
tcrB_test = enc_list_bl_max_len(test_data.B3, blosum50_20aa, 23)
pep_test = enc_list_bl_max_len(test_data.peptide, blosum50_20aa,15)
y_test = np.array(test_data.Binding)

print(f"tcrB_test shape: {tcrB_test.shape}")
print(f"tcrB_train shape: {tcrB_train.shape}")
print(f"pep_test shape: {pep_test.shape}")
print(f"pep_train shape: {pep_train.shape}")


# Network architecture
def CNN_extra():
    cdrb_in = Input(shape=(23, 20))
    pep_in = Input(shape=(15, 20))

    cdrb_conv1 = Conv1D(16, 1, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
    cdrb_pool1 = GlobalMaxPooling1D()(cdrb_conv1)
    cdrb_conv3 = Conv1D(16, 3, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
    cdrb_pool3 = GlobalMaxPooling1D()(cdrb_conv3)
    cdrb_conv5 = Conv1D(16, 5, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
    cdrb_pool5 = GlobalMaxPooling1D()(cdrb_conv5)
    cdrb_conv7 = Conv1D(16, 7, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
    cdrb_pool7 = GlobalMaxPooling1D()(cdrb_conv7)
    cdrb_conv9 = Conv1D(16, 9, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
    cdrb_pool9 = GlobalMaxPooling1D()(cdrb_conv9)

    pep_conv1 = Conv1D(16, 1, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
    pep_pool1 = GlobalMaxPooling1D()(pep_conv1)
    pep_conv3 = Conv1D(16, 3, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
    pep_pool3 = GlobalMaxPooling1D()(pep_conv3)
    pep_conv5 = Conv1D(16, 5, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
    pep_pool5 = GlobalMaxPooling1D()(pep_conv5)
    pep_conv7 = Conv1D(16, 7, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
    pep_pool7 = GlobalMaxPooling1D()(pep_conv7)
    pep_conv9 = Conv1D(16, 9, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
    pep_pool9 = GlobalMaxPooling1D()(pep_conv9)

    cdr3_cat = concatenate([cdrb_pool1, cdrb_pool3, cdrb_pool5, cdrb_pool7, cdrb_pool9], name='cdr3_cat')
    pep_cat = concatenate([pep_pool1, pep_pool3, pep_pool5, pep_pool7, pep_pool9], name='pep_cat')
    cat = concatenate([cdr3_cat, pep_cat], axis=1)

    dense = Dense(32, activation='sigmoid')(cat)
    out = Dense(1, activation='sigmoid')(dense)
    model = (Model(inputs=[cdrb_in, pep_in], outputs=[out]))
    return model
# Call and compile the model
model = CNN_extra()
model.compile(loss="binary_crossentropy",
              optimizer=adam_v2.Adam(learning_rate=lr),
              metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(curve="ROC", name="ROCAUC"),tf.keras.metrics.Precision(name="Precision"),
        tf.keras.metrics.Recall(name="Recall"),
        tf.keras.metrics.AUC(curve="PR", name="F1")  # F1-score can be approximated using the PR curve
    ]
)

# Train model
print('Training..')
if not os.path.exists(root_model):
    os.makedirs(root_model)
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)
early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=0, mode='min', restore_best_weights=True)
checkpointer = keras.callbacks.ModelCheckpoint(
    os.path.join(save_model_path, 'CNN_extra_feature_epoch{epoch:03d}_AUC{val_ROCAUC:.6f}.hdf5'),
    save_weights_only=True)
# history = model.fit([tcrb_train, pep_train], y_train, validation_data=([tcrb_test, pep_test], y_test), epochs=epochs,
#                   batch_size=batchsize, verbose=1, callbacks=[early_stop, checkpointer])
train_inputs = [tcrB_train, pep_train]
validation_inputs = [tcrB_test, pep_test]
history = model.fit(train_inputs, y_train,
                    validation_data=(validation_inputs, y_test),
                    epochs=epochs, batch_size=batchsize, verbose=1, callbacks=[early_stop, checkpointer])


# Logs of results从一个训练历史对象（通常是由Keras模型训练过程中产生的）中提取关键指标（如损失、准确率、AUC等），并将这些指标以及对应的轮次（epoch）存储到一个Pandas DataFrame中，最后将这个DataFrame保存为一个TSV（制表符分隔值）文件
history_dict = history.history  # 从history对象中提取所有训练过程中的历史记录，并将其存储在history_dict字典中。history对象是在使用Keras训练模型时，通过model.fit()方法返回的，它包含了训练过程中的损失值、准确率等指标的记录。
epoch = []  # 初始化空列表epoch用于存储轮次
loss = history_dict['loss']  # 从history_dict字典中提取关键指标
acc = history_dict['binary_accuracy']
auc_roc = history_dict['ROCAUC']
precision = history_dict['Precision']
recall = history_dict['Recall']
f1_score = history_dict['F1']
val_loss = history_dict['val_loss']
val_acc = history_dict['val_binary_accuracy']
val_auc_roc = history_dict['val_ROCAUC']
val_precision = history_dict['val_Precision']
val_recall = history_dict['val_Recall']
val_f1_score = history_dict['val_F1']

for ep in range(len(acc)):  # 遍历了训练准确率的长度（即训练的轮次数）
    epoch.append(ep + 1)  # 训练过程的第一轮次称为第1轮次，而不是第0轮次。循环中的ep+1就是实现了这个目的，它将循环变量ep（从0开始）加1

dfhistory = {'epoch': epoch,
    'loss': loss, 'acc': acc, 'auc_roc': auc_roc,
    'precision': precision, 'recall': recall, 'f1_score': f1_score,
    'val_loss': val_loss, 'val_acc': val_acc, 'val_auc_roc': val_auc_roc,
    'val_precision': val_precision, 'val_recall': val_recall, 'val_f1_score': val_f1_score}  # 创建了一个字典dfhistory，它包含了所有要保存到DataFrame中的列名和对应的数据。列名包括'epoch'、'loss'、'acc'、'auc_roc'、'val_loss'、'val_acc'和'val_auc_roc'
df = pd.DataFrame(
    dfhistory)  # 使用Pandas库的DataFrame构造函数，将dfhistory字典转换为DataFrame对象df。DataFrame是Pandas中用于存储和操作结构化数据的主要数据结构
if not os.path.exists(
        root_history):  # os.path.exists()用于检查root_history路径是否存在，不存在，则创建这个路径。这是为了确保在保存文件之前，目标文件夹已经存在。os.makedirs()用于创建多级目录
    os.makedirs(root_history)
df.to_csv(os.path.join(root_history, "CNN_extra_feature_{}.tsv".format(model_dir2)), header=True, sep='\t',
          index=False)  # 将DataFramedf保存为一个TSV文件。文件路径是通过将root_history路径和文件名（通过格式化字符串生成，文件名中包含model_dir2变量）组合而成的。header=True表示文件中包含列名作为表头，sep='\t'指定字段分隔符为制表符（Tab），index=False表示不将DataFrame的索引保存到文件中

# get num_epoch从一个Pandas DataFrame (df) 中提取验证集的AUC-ROC值和验证损失，并找出这些指标分别在何时达到最优（即AUC-ROC最大时和验证损失最小时）的轮次
val_auc_roc = list(df['val_auc_roc'])  # 将DataFrame中'val_auc_roc'列的值转换为列表
max_auc_index = val_auc_roc.index(max(val_auc_roc))  # 找出验证集AUC-ROC值最大的索引,这个索引位置对应的就是模型在训练过程中达到最大AUC-ROC值的轮次。
num_epoch = max_auc_index + 1  # 计算达到最大AUC-ROC值的轮次（加1是因为通常轮次从1开始计数）
print("max auc epoch:", num_epoch)

val_loss = list(df['val_loss'])
min_val_loss_index = val_loss.index(min(val_loss))  # 找出验证损失最小的索引
loss_num_epoch = min_val_loss_index + 1  # 计算达到最小验证损失值的轮次
print("loss auc epoch:", loss_num_epoch)

# extra CNN feature
for root, dirs, files in os.walk(save_model_path):
    for file in files:
        file_path = os.path.join(save_model_path, file)
        if file.startswith("CNN_extra_feature_epoch{:03d}".format(num_epoch)):
            PATH = file_path
            model.load_weights(PATH, by_name=True)
            cat_cdr3_model = Model(inputs=model.input, outputs=model.get_layer('cdr3_cat').output)
            cat_peptide_model = Model(inputs=model.input, outputs=model.get_layer('pep_cat').output)

            # 使用所有输入来预测
            train_inputs_full = [tcrB_train, pep_train]
            test_inputs_full = [tcrB_test, pep_test]

            cat_cdr3_output_train = cat_cdr3_model.predict(train_inputs_full)
            cat_peptide_output_train = cat_peptide_model.predict(train_inputs_full)

            cat_cdr3_output_test = cat_cdr3_model.predict(test_inputs_full)
            cat_peptide_output_test = cat_peptide_model.predict(test_inputs_full)

            # cat_cdr3_output_train = cat_cdr3_model.predict([tcrB_train, tcra_train, pep_train])
            # cat_peptide_output_train = cat_peptide_model.predict([[tcrB_train, tcra_train, pep_train]])
            #
            # cat_cdr3_output_test = cat_cdr3_model.predict([tcrB_test, tcra_test, pep_test])
            # cat_peptide_output_test = cat_peptide_model.predict([tcrB_test, tcra_test, pep_test])

            # Define the new file name
            new_file_name = 'max_AUC_' + file
            new_file_path = os.path.join(save_model_path, new_file_name)

            # Check if the target file already exists and handle accordingly
            if os.path.exists(new_file_path):
                os.remove(new_file_path)

            os.rename(file_path, new_file_path)

        elif file.startswith("CNN_extra_feature_epoch{:03d}".format(loss_num_epoch)):
            # Define the new file name
            new_file_name = 'minloss_AUC_' + file
            new_file_path = os.path.join(save_model_path, new_file_name)

            # Check if the target file already exists and handle accordingly
            if os.path.exists(new_file_path):
                os.remove(new_file_path)

            os.rename(file_path, new_file_path)

        else:
            # Remove files that do not match the criteria
            os.remove(file_path)

cdr3b_dict_train = {}
for i in range(len(train_data.B3)):
    if train_data.B3[i] not in cdr3b_dict_train.keys():
        cdr3b_dict_train[train_data.B3[i]] = cat_cdr3_output_train[i]
peptide_dict_train = {}
for i in range(len(train_data.peptide)):
    if train_data.peptide[i] not in peptide_dict_train.keys():
        peptide_dict_train[train_data.peptide[i]] = cat_peptide_output_train[i]

cdr3b_dict_test = {}
for i in range(len(test_data.B3)):
    if test_data.B3[i] not in cdr3b_dict_test.keys():
        cdr3b_dict_test[test_data.B3[i]] = cat_cdr3_output_test[i]
peptide_dict_test = {}
for i in range(len(test_data.peptide)):
    if test_data.peptide[i] not in peptide_dict_test.keys():
        peptide_dict_test[test_data.peptide[i]] = cat_peptide_output_test[i]

with open(train_path_pickle_cdr3, 'wb') as f1:
    pickle.dump(cdr3b_dict_train, f1)
with open(train_path_pickle_peptide, 'wb') as f2:
    pickle.dump(peptide_dict_train, f2)
print("train dataset CNN feature has saved!")

with open(test_path_pickle_cdr3, 'wb') as f3:
    pickle.dump(cdr3b_dict_test, f3)
with open(test_path_pickle_peptide, 'wb') as f4:
    pickle.dump(peptide_dict_test, f4)
print("test dataset CNN feature has saved!")