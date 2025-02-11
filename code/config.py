

from argparse import ArgumentParser

#Args parser 
parser = ArgumentParser(description="Specifying Input Parameters")

# Data directory
parser.add_argument("-pd", "--pridir", default="../data", help="Primary directory of data")#数据的一级目录

# parser.add_argument("-sd", "--secdir", default="nettcrfull5folds", help="Secondary directory of data")#数据二级目录，13种肽
parser.add_argument("-sd", "--secdir", default="nettcr_bal5folds", help="Secondary directory of data")#数据二级目录，13种肽
# parser.add_argument("-sd", "--secdir", default="nettcr_strict0_5folds", help="Secondary directory of data")#数据二级目录，13种肽

parser.add_argument("-td", "--terdir", default="fold0", help="Tertiary directory of data")#数据三级目录
parser.add_argument("-td2", "--terdir2", default="predresult", help="Tertiary directory of data")#数据三级目录

# 添加 testdelimiter 参数
parser.add_argument("-tdel", "--testdelimiter", default="\t", help="Delimiter for test data")


# Parameter setting
parser.add_argument("-e", "--epochs", default=200, type=int, help="Number of training epochs")#允许用户指定训练模型的周期数（即遍历整个训练数据集的次数）。如果未指定，则默认为1000个周期。参数类型为整数（int），并附带了帮助信息（help），说明了这个参数的作用。


parser.add_argument("-bs", "--batchsize", default=256, type=int, help="Batch size of neural network")#定义了批处理大小（即每次训练时输入到模型中的样本数）的命令行参数。用户可以通过-bs或--batchsize来指定它，默认值为512。参数类型为整数，并附带了帮助信息。
parser.add_argument("-cnnlr", "--cnnlearningrate", default=0.001, type=float, help="Learning rate of CNN extra feature Network")#定义了CNN（卷积神经网络）特征网络的学习率参数。用户可以通过-cnnlr或--cnnlearningrate来指定它，默认值为0.001。参数类型为浮点数（float），并附带了相应的帮助信息。
parser.add_argument("-gnnlr", "--gnnlearningrate", default=0.0001, type=float, help="Learning rate of HeteroTCR")#定义了HeteroTCR（可能是某种图神经网络模型）的学习率参数。用户可以通过-gnnlr或--gnnlearningrate来指定它，默认值为0.0001。同样，参数类型为浮点数，并附带了帮助信息
parser.add_argument("-wd", "--weightdecay", default=0, type=float, help="Weight decay of HeteroTCR")#义了HeteroTCR网络的权重衰减参数，用于防止过拟合。用户可以通过-wd或--weightdecay来指定它，默认值为0（即不使用权重衰减）。参数类型为浮点数，并附带了帮助信息。
parser.add_argument("-hc", "--hiddenchannels", default=1024, type=int, help="Number of hidden channels")#定义了网络隐藏层的通道数（或称为隐藏单元数）。用户可以通过-hc或--hiddenchannels来指定它，默认值为1024。参数类型为整数，并附带了帮助信息。
parser.add_argument("-nl", "--numberlayers", default=3, type=int, help="Number of layers")#定义了网络层数的命令行参数。用户可以通过-nl或--numberlayers来指定它，默认值为3层。参数类型为整数，并附带了帮助信息。
parser.add_argument("-nt", "--gnnnet", default="SAGE", help="Network type of GNN (SAGE | TF | FiLM)")#定义了GNN（图神经网络）网络类型的命令行参数。用户可以通过-net或--gnnnet来指定它，默认值为SAGE。这个参数不限制为整数或浮点数，而是字符串类型，因为这里是在选择一种网络架构。帮助信息列出了可选的网络类型（SAGE、TF、FiLM）。
parser.add_argument("-dropout", "--dropout_rate", default=0.5, type=float,help="dropout")#定义了GNN（图神经网络）网络类型的命令行参数。用户可以通过-net或--gnnnet来指定它，默认值为SAGE。这个参数不限制为整数或浮点数，而是字符串类型，因为这里是在选择一种网络架构。帮助信息列出了可选的网络类型（SAGE、TF、FiLM）。
# parser.add_argument("-similarity_dim", "--similarity_dim", default=1844, type=int, help="similarity_dim")  # 添加这个参数，没有用到

# Models & History save directory
parser.add_argument("-md", "--modeldir", default="../model", help="Primary directory of models save directory")#定义了一个命令行参数-md或--modeldir，用于指定模型保存的主要目录。如果用户没有通过命令行明确指定这个参数，那么模型将被保存到默认的../model目录下。这个路径是相对于当前工作目录的父目录中的model文件夹。
parser.add_argument("-hd", "--hisdir", default="../History", help="Primary directory of history save directory")#定义了一个命令行参数-hd或--hisdir，用于指定训练历史记录（如损失值、准确率等）保存的主要目录。如果用户没有指定，则默认保存到../History目录下。这个路径同样是相对于当前工作目录的父目录中的History文件夹。
# parser.add_argument("-tmd", "--testmodeldir", default="publicTCRs2_5folds_fold3_HeteroAB_New2", help="Secondary directory of test model directory")#定义了一个命令行参数-tmd或--testmodeldir，用于指定测试模型目录中的次级目录名。这个参数可能用于进一步组织测试模型文件，例如在进行交叉验证或不同实验设置时区分模型。如果用户没有指定，则默认使用iedb_5folds_fold0_Hetero作为次级目录名。这个参数的值并不直接指定一个文件路径，而是用于在模型保存的主要目录（由--modeldir指定）下创建或指定一个子目录。
#指定一个目录路径，该路径用于指向测试模型的次级目录

# parser.add_argument("-tmd", "--testmodeldir", default="nettcrfull5folds_fold0_GuoAndXiacaiyang_HeteroAB", help="Secondary directory of test model directory")#定义了一个命令行参数-tmd或--testmodeldir，用于指定测试模型目录中的次级目录名。这个参数可能用于进一步组织测试模型文件，例如在进行交叉验证或不同实验设置时区分模型。如果用户没有指定，则默认使用iedb_5folds_fold0_Hetero作为次级目录名。这个参数的值并不直接指定一个文件路径，而是用于在模型保存的主要目录（由--modeldir指定）下创建或指定一个子目录。
# parser.add_argument("-tmd", "--testmodeldir", default="nettcr_bal5folds_fold0_GuoAndXiacaiyang_HeteroAB", help="Secondary directory of test model directory")#定义了一个命令行参数-tmd或--testmodeldir，用于指定测试模型目录中的次级目录名。这个参数可能用于进一步组织测试模型文件，例如在进行交叉验证或不同实验设置时区分模型。如果用户没有指定，则默认使用iedb_5folds_fold0_Hetero作为次级目录名。这个参数的值并不直接指定一个文件路径，而是用于在模型保存的主要目录（由--modeldir指定）下创建或指定一个子目录。
parser.add_argument("-tmd", "--testmodeldir", default="nettcr_bal5folds_fold0_GuoAndXiacaiyang_CNNAB", help="Secondary directory of test model directory")#定义了一个命令行参数-tmd或--testmodeldir，用于指定测试模型目录中的次级目录名。这个参数可能用于进一步组织测试模型文件，例如在进行交叉验证或不同实验设置时区分模型。如果用户没有指定，则默认使用iedb_5folds_fold0_Hetero作为次级目录名。这个参数的值并不直接指定一个文件路径，而是用于在模型保存的主要目录（由--modeldir指定）下创建或指定一个子目录。

# parser.add_argument("-tmd", "--testmodeldir", default="nettcr_bal5folds_fold2_SingleGraph", help="Secondary directory of test model directory")#定义了一个命令行参数-tmd或--testmodeldir，用于指定测试模型目录中的次级目录名。这个参数可能用于进一步组织测试模型文件，例如在进行交叉验证或不同实验设置时区分模型。如果用户没有指定，则默认使用iedb_5folds_fold0_Hetero作为次级目录名。这个参数的值并不直接指定一个文件路径，而是用于在模型保存的主要目录（由--modeldir指定）下创建或指定一个子目录。
# parser.add_argument("-tmd", "--testmodeldir", default="nettcr_strict0_5folds_fold0_SingleGraph", help="Secondary directory of test model directory")#定义了一个命令行参数-tmd或--testmodeldir，用于指定测试模型目录中的次级目录名。这个参数可能用于进一步组织测试模型文件，例如在进行交叉验证或不同实验设置时区分模型。如果用户没有指定，则默认使用iedb_5folds_fold0_Hetero作为次级目录名。这个参数的值并不直接指定一个文件路径，而是用于在模型保存的主要目录（由--modeldir指定）下创建或指定一个子目录。

# parser.add_argument("-tmd", "--testmodeldir", default="nettcr_strict0_5folds_fold0_GuoAndXiacaiyang_HeteroAB", help="Secondary directory of test model directory")#定义了一个命令行参数-tmd或--testmodeldir，用于指定测试模型目录中的次级目录名。这个参数可能用于进一步组织测试模型文件，例如在进行交叉验证或不同实验设置时区分模型。如果用户没有指定，则默认使用iedb_5folds_fold0_Hetero作为次级目录名。这个参数的值并不直接指定一个文件路径，而是用于在模型保存的主要目录（由--modeldir指定）下创建或指定一个子目录。
# parser.add_argument("-tmd", "--testmodeldir", default="nettcr_strict0_5folds_fold0_GuoAndXiacaiyang_CNNAB", help="Secondary directory of test model directory")#定义了一个命令行参数-tmd或--testmodeldir，用于指定测试模型目录中的次级目录名。这个参数可能用于进一步组织测试模型文件，例如在进行交叉验证或不同实验设置时区分模型。如果用户没有指定，则默认使用iedb_5folds_fold0_Hetero作为次级目录名。这个参数的值并不直接指定一个文件路径，而是用于在模型保存的主要目录（由--modeldir指定）下创建或指定一个子目录。

# parser.add_argument("-tmd", "--testmodeldir", default="nettcrfull5folds_fold4_Base_HeteroAB", help="Secondary directory of test model directory")#定义了一个命令行参数-tmd或--testmodeldir，用于指定测试模型目录中的次级目录名。这个参数可能用于进一步组织测试模型文件，例如在进行交叉验证或不同实验设置时区分模型。如果用户没有指定，则默认使用iedb_5folds_fold0_Hetero作为次级目录名。这个参数的值并不直接指定一个文件路径，而是用于在模型保存的主要目录（由--modeldir指定）下创建或指定一个子目录。
# parser.add_argument("-tmd", "--testmodeldir", default="nettcr_bal5folds_fold4_Base_HeteroAB", help="Secondary directory of test model directory")#定义了一个命令行参数-tmd或--testmodeldir，用于指定测试模型目录中的次级目录名。这个参数可能用于进一步组织测试模型文件，例如在进行交叉验证或不同实验设置时区分模型。如果用户没有指定，则默认使用iedb_5folds_fold0_Hetero作为次级目录名。这个参数的值并不直接指定一个文件路径，而是用于在模型保存的主要目录（由--modeldir指定）下创建或指定一个子目录。
# parser.add_argument("-tmd", "--testmodeldir", default="nettcr_strict4_5folds_fold4_Base_HeteroAB", help="Secondary directory of test model directory")#定义了一个命令行参数-tmd或--testmodeldir，用于指定测试模型目录中的次级目录名。这个参数可能用于进一步组织测试模型文件，例如在进行交叉验证或不同实验设置时区分模型。如果用户没有指定，则默认使用iedb_5folds_fold0_Hetero作为次级目录名。这个参数的值并不直接指定一个文件路径，而是用于在模型保存的主要目录（由--modeldir指定）下创建或指定一个子目录。





# cuda
parser.add_argument("-cu", "--cuda", default="0", help="Number of gpu device")
args = parser.parse_args()
