

from argparse import ArgumentParser

#Args parser 
parser = ArgumentParser(description="Specifying Input Parameters")

# Data directory
parser.add_argument("-pd", "--pridir", default="../data", help="Primary directory of data")

# parser.add_argument("-sd", "--secdir", default="nettcrfull5folds", help="Secondary directory of data")
parser.add_argument("-sd", "--secdir", default="nettcr_bal5folds", help="Secondary directory of data")
# parser.add_argument("-sd", "--secdir", default="nettcr_strict0_5folds", help="Secondary directory of data")

parser.add_argument("-td", "--terdir", default="fold0", help="Tertiary directory of data")
parser.add_argument("-td2", "--terdir2", default="predresult", help="Tertiary directory of data")


parser.add_argument("-tdel", "--testdelimiter", default="\t", help="Delimiter for test data")


# Parameter setting
parser.add_argument("-e", "--epochs", default=200, type=int, help="Number of training epochs")

parser.add_argument("-bs", "--batchsize", default=256, type=int, help="Batch size of neural network")
parser.add_argument("-cnnlr", "--cnnlearningrate", default=0.001, type=float, help="Learning rate of CNN extra feature Network")
parser.add_argument("-gnnlr", "--gnnlearningrate", default=0.0001, type=float, help="Learning rate of HeteroTCR")
parser.add_argument("-wd", "--weightdecay", default=0, type=float, help="Weight decay of HeteroTCR")
parser.add_argument("-hc", "--hiddenchannels", default=1024, type=int, help="Number of hidden channels")
parser.add_argument("-nl", "--numberlayers", default=3, type=int, help="Number of layers")
parser.add_argument("-nt", "--gnnnet", default="SAGE", help="Network type of GNN (SAGE | TF | FiLM)")
parser.add_argument("-dropout", "--dropout_rate", default=0.5, type=float,help="dropout")
# parser.add_argument("-similarity_dim", "--similarity_dim", default=1844, type=int, help="similarity_dim") 

# Models & History save directory
parser.add_argument("-md", "--modeldir", default="../model", help="Primary directory of models save directory")
parser.add_argument("-hd", "--hisdir", default="../History", help="Primary directory of history save directory")
# parser.add_argument("-tmd", "--testmodeldir", default="publicTCRs2_5folds_fold3_HeteroAB_New2", help="Secondary directory of test model directory")

# parser.add_argument("-tmd", "--testmodeldir", default="nettcrfull5folds_fold0_GuoAndXiacaiyang_HeteroAB", help="Secondary directory of test model directory")
# parser.add_argument("-tmd", "--testmodeldir", default="nettcr_bal5folds_fold0_GuoAndXiacaiyang_HeteroAB", help="Secondary directory of test model directory")
parser.add_argument("-tmd", "--testmodeldir", default="nettcr_bal5folds_fold0_GuoAndXiacaiyang_CNNAB", help="Secondary directory of test model directory")
# parser.add_argument("-tmd", "--testmodeldir", default="nettcr_bal5folds_fold2_SingleGraph", help="Secondary directory of test model directory")
# parser.add_argument("-tmd", "--testmodeldir", default="nettcr_strict0_5folds_fold0_SingleGraph", help="Secondary directory of test model directory")
# parser.add_argument("-tmd", "--testmodeldir", default="nettcr_strict0_5folds_fold0_GuoAndXiacaiyang_HeteroAB", help="Secondary directory of test model directory")
# parser.add_argument("-tmd", "--testmodeldir", default="nettcr_strict0_5folds_fold0_GuoAndXiacaiyang_CNNAB", help="Secondary directory of test model directory")
# parser.add_argument("-tmd", "--testmodeldir", default="nettcrfull5folds_fold4_Base_HeteroAB", help="Secondary directory of test model directory")
# parser.add_argument("-tmd", "--testmodeldir", default="nettcr_strict4_5folds_fold4_Base_HeteroAB", help="Secondary directory of test model directory")



# cuda
parser.add_argument("-cu", "--cuda", default="0", help="Number of gpu device")
args = parser.parse_args()
