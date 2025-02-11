

from argparse import ArgumentParser

#Args parser
parser = ArgumentParser(description="Specifying Input Parameters")

# Data directory
parser.add_argument("-pd", "--pridir", default="../data", help="Primary directory of data")
# parser.add_argument("-sd", "--secdir", default="nettcr_bal5folds", help="Secondary directory of data")
# parser.add_argument("-sd", "--secdir", default="nettcrfull5folds", help="Secondary directory of data")
# parser.add_argument("-sd", "--secdir", default="nettcr_strict4_5folds", help="Secondary directory of data")#数据二级目录，13种肽
parser.add_argument("-sd", "--secdir", default="nettcr_strict4_5folds", help="Secondary directory of data")#数据二级目录，13种肽


parser.add_argument("-td", "--terdir", default="fold4", help="Tertiary directory of data")
parser.add_argument("-td2", "--terdir2", default="predresult", help="Tertiary directory of data")#数据三级目录

# Parameter setting
parser.add_argument("-e", "--epochs", default=1000, type=int, help="Number of training epochs")
parser.add_argument("-bs", "--batchsize", default=512, type=int, help="Batch size of neural network")
parser.add_argument("-cnnlr", "--cnnlearningrate", default=0.001, type=float, help="Learning rate of CNN extra feature Network")
parser.add_argument("-gnnlr", "--gnnlearningrate", default=0.0001, type=float, help="Learning rate of HeteroTCR")
parser.add_argument("-wd", "--weightdecay", default=0, type=float, help="Weight decay of HeteroTCR")
parser.add_argument("-hc", "--hiddenchannels", default=1024, type=int, help="Number of hidden channels")
parser.add_argument("-nl", "--numberlayers", default=3, type=int, help="Number of layers")
parser.add_argument("-net", "--gnnnet", default="SAGE", help="Network type of GNN (SAGE | TF | FiLM)")
# 添加 testdelimiter 参数
parser.add_argument("-tdel", "--testdelimiter", default="\t", help="Delimiter for test data")

# Models & History save directory
parser.add_argument("-md", "--modeldir", default="../model", help="Primary directory of models save directory")
parser.add_argument("-hd", "--hisdir", default="../History", help="Primary directory of history save directory")
# parser.add_argument("-tmd", "--testmodeldir", default="nettcr_bal5folds_fold4_Graph", help="Secondary directory of test model directory")
# parser.add_argument("-tmd", "--testmodeldir", default="nettcrfull5folds_fold4_Graph", help="Secondary directory of test model directory")
parser.add_argument("-tmd", "--testmodeldir", default="nettcr_strict4_5folds_fold4_Graph", help="Secondary directory of test model directory")#定义了一个命令行参数-tmd或--testmodeldir，用于指定测试模型目录中的次级目录名。这个参数可能用于进一步组织测试模型文件，例如在进行交叉验证或不同实验设置时区分模型。如果用户没有指定，则默认使用iedb_5folds_fold0_Hetero作为次级目录名。这个参数的值并不直接指定一个文件路径，而是用于在模型保存的主要目录（由--modeldir指定）下创建或指定一个子目录。


# cuda
parser.add_argument("-cu", "--cuda", default="0", help="Number of gpu device")

args = parser.parse_args()