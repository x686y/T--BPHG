

import os, sys
from tensorflow.keras.layers import BatchNormalization, Dropout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras_metrics as km
# import keras.metrics as km
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, concatenate, Flatten, Embedding ,Dropout,Layer
from keras.optimizers import adam_v2
from keras.initializers import glorot_normal
from keras.activations import sigmoid
from keras.callbacks import EarlyStopping
import random
import pickle
from config import *

def seed_tensorflow(seed=42):
    random.seed(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed) 
    tf.random.set_seed(seed)  


seed_tensorflow()  


epochs = int(args.epochs) 
batchsize = int(args.batchsize)
lr = float(args.cnnlearningrate)

# Path construction
root = os.path.join(args.pridir, args.secdir, args.terdir) 
train_csv = os.path.join(root, 'train_data.tsv') 
test_csv = os.path.join(root, 'test_data.tsv')

root_model = args.modeldir 
model_dir = str(args.secdir) + '_' + str(
    args.terdir) + '_BaseCNNAB' 
model_dir2 = str(args.secdir) + '_' + str(args.terdir)
save_model_path = os.path.join(root_model, model_dir)  
root_history = args.hisdir  

#
train_path_pickle_cdr3= os.path.join(root,'BaseCNN_feature_cdr3_train.pickle') 
train_path_pickle_peptidecdr3a= os.path.join(root, 'BaseCNN_feature_peptidecdr3a_train.pickle')
train_path_pickle_peptidecdr3b  = os.path.join(root, 'BaseCNN_feature_peptidecdr3b_train.pickle')
test_path_pickle_cdr3 = os.path.join(root, 'BaseCNN_feature_cdr3_test.pickle')
test_path_pickle_peptidecdr3a = os.path.join(root, 'BaseCNN_feature_peptidecdr3a_test.pickle')
test_path_pickle_peptidecdr3b = os.path.join(root, 'BaseCNN_feature_peptidecdr3b_test.pickle')


# Load data
print('Loading the train data..')
train_data = pd.read_csv(train_csv,
                         delimiter='\t')  
print(f"Length of train_data: {len(train_data)}") 

print('Loading the test data..')
test_data = pd.read_csv(test_csv, delimiter='\t')
print(f"Length of test_data: {len(test_data)}")  
print(train_data.columns)
print(train_data.info())

def enc_list_bl_max_len(aa_seqs, blosum,
                        max_seq_len):  
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
    sequences = []  
    for seq in aa_seqs: 
        e_seq = np.zeros((len(seq), len(
            blosum["A"]))) 
        count = 0  
        for aa in seq: 
            if aa in blosum: 
                e_seq[count] = blosum[aa]  
                count += 1  
            else: 
                print(seq) 
                sys.stderr.write(
                    "Unknown amino acid in peptides: " + aa + ", encoding aborted!\n") 
                sys.exit(2)  
        sequences.append(e_seq)  

    # pad sequences
    n_seqs = len(aa_seqs) 
    n_features = sequences[0].shape[1] 
    enc_aa_seq = np.zeros((n_seqs, max_seq_len, n_features)) 
    for i in range(0, n_seqs): 
        enc_aa_seq[i, :sequences[i].shape[0], :n_features] = sequences[
            i] 
    return enc_aa_seq  

blosum62_20aa = {
    'A': np.array((4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0)),
    'R': np.array((-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3)),
    'N': np.array((-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -3, 1, 0, -4, -2, -3)),
    'D': np.array((-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3)),
    'C': np.array((0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1)),
    'Q': np.array((-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -2, -1, 0, -1, -2, -1, -2)),
    'E': np.array((-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -3)),
    'G': np.array((0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3)),
    'H': np.array((-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3)),
    'I': np.array((-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -3, -1, -3, -1, 3)),
    'L': np.array((-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -3, -1, -2, -1, 1)),
    'K': np.array((-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2)),
    'M': np.array((-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -2, -1, -1, -1, 1)),
    'F': np.array((-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1)),
    'P': np.array((-1, -2, -3, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2)),
    'S': np.array((1, -1, 1, 0, -1, 0, 0, 0, -1, -3, -3, 0, -2, -2, -1, 4, 1, -3, -2, -2)),
    'T': np.array((0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0)),
    'W': np.array((-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3)),
    'Y': np.array((-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1)),
    'V': np.array((0, -3, -3, -3, -1, -2, -3, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4))
}

# Feature encoding
print('Encoding the train data..')
tcrB_train = enc_list_bl_max_len(train_data.B3, blosum62_20aa, 40)
tcra_train = enc_list_bl_max_len(train_data.A3, blosum62_20aa, 40)
pep_train = enc_list_bl_max_len(train_data.peptide, blosum62_20aa, 40)  
y_train = np.array(train_data.Binding)  

print('Encoding the test data..')

tcrB_test = enc_list_bl_max_len(test_data.B3, blosum62_20aa, 40)
tcra_test = enc_list_bl_max_len(test_data.A3, blosum62_20aa, 40)

# tcrb_test = np.concatenate((tcrB_test, tcra_test), axis=1)
pep_test = enc_list_bl_max_len(test_data.peptide, blosum62_20aa, 40)
y_test = np.array(test_data.Binding)

print(f"tcrB_test shape: {tcrB_test.shape}")

print(f"tcra_train shape: {tcra_train.shape}")
print(f"tcrB_train shape: {tcrB_train.shape}")

print(f"tcra_test shape: {tcra_test.shape}")
print(f"pep_test shape: {pep_test.shape}")
print(f"pep_train shape: {pep_train.shape}")
class MultiHeadAttention(Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"

        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output, attention_weights

    def separate_heads(self, x, batch_size):
        """
       Split the input tensor into multiple heads
        """
        seq_len = x.shape[1]  
        if seq_len is None:
            seq_len = tf.shape(x)[1]  
       
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.head_dim))
        # print("x shape after reshape:", x.shape)
        return tf.transpose(x, perm=[0, 2, 1, 3])  # [batch_size, num_heads, seq_len, head_dim]

    def combine_heads(self, x):

        x = tf.transpose(x, perm=[0, 2, 1, 3])  # [batch_size, seq_len, num_heads, head_dim]
        seq_len = x.shape[1] 
        if seq_len is None:
            seq_len = tf.shape(x)[1]  

       
        x = tf.reshape(x, (tf.shape(x)[0], seq_len, self.embed_dim))  # [batch_size, seq_len, embed_dim]
        x.set_shape((None, seq_len, self.embed_dim))  
        return x

    def call(self, inputs, mask=None):
        query, key, value = inputs
        batch_size = tf.shape(query)[0]

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(query, key, value, mask)

       
        scaled_attention = self.combine_heads(scaled_attention)
        return scaled_attention, attention_weights



def multi_head_attention_layer(query, key, value, embed_dim, num_heads):
    mha = MultiHeadAttention(embed_dim, num_heads)
    attention_output, weights = mha([query, key, value])
    return attention_output 
def CNN_extra():
    cdra_in = Input(shape=(40, 20), name='cdra_in')
    cdrb_in = Input(shape=(40, 20), name='cdrb_in')
    pep_in = Input(shape=(40, 20), name='pep_in')

    def conv_block(x, filters, kernel_size, name):
        x = Conv1D(filters, kernel_size, padding='same', activation='relu', kernel_initializer='glorot_normal',
                   name=name)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)  # Dropout
        return x

    cdr_conv1 = conv_block(cdra_in, 16, 1, 'cdr_conv1')
    cdr_conv3 = conv_block(cdra_in, 16, 3, 'cdr_conv3')
    cdr_conv5 = conv_block(cdra_in, 16, 5, 'cdr_conv5')
    cdr_conv7 = conv_block(cdra_in, 16, 7, 'cdr_conv7')
    cdr_conv9 = conv_block(cdra_in, 16, 9, 'cdr_conv9')

    cdrb_conv1 = conv_block(cdrb_in, 16, 1, 'cdrb_conv1')
    cdrb_conv3 = conv_block(cdrb_in, 16, 3, 'cdrb_conv3')
    cdrb_conv5 = conv_block(cdrb_in, 16, 5, 'cdrb_conv5')
    cdrb_conv7 = conv_block(cdrb_in, 16, 7, 'cdrb_conv7')
    cdrb_conv9 = conv_block(cdrb_in, 16, 9, 'cdrb_conv9')

    pep_conv1 = conv_block(pep_in, 16, 1, 'pep_conv1')
    pep_conv3 = conv_block(pep_in, 16, 3, 'pep_conv3')
    pep_conv5 = conv_block(pep_in, 16, 5, 'pep_conv5')
    pep_conv7 = conv_block(pep_in, 16, 7, 'pep_conv7')
    pep_conv9 = conv_block(pep_in, 16, 9, 'pep_conv9')

 
    cdr3a_cat = concatenate([cdr_conv1, cdr_conv3, cdr_conv5, cdr_conv7, cdr_conv9], name='cdr3a_cat')
    cdr3b_cat = concatenate([cdrb_conv1, cdrb_conv3, cdrb_conv5, cdrb_conv7, cdrb_conv9], name='cdr3b_cat')
    pep_cat = concatenate([pep_conv1, pep_conv3, pep_conv5, pep_conv7, pep_conv9], name='pep_cat')


    self_attention_cdr3a = multi_head_attention_layer(
        cdr3a_cat, cdr3a_cat, cdr3a_cat, embed_dim=128, num_heads=8)
    print("self_attention_cdr3a shape:", self_attention_cdr3a.shape)
    self_attention_cdr3b = multi_head_attention_layer(
        cdr3b_cat, cdr3b_cat, cdr3b_cat, embed_dim=128, num_heads=8)
    print("self_attention_cdr3b shape:", self_attention_cdr3b.shape)
    self_attention_pep_cat = multi_head_attention_layer(
        pep_cat, pep_cat, pep_cat, embed_dim=128, num_heads=8)
    print("self_attention_pep_cat shape:", self_attention_pep_cat.shape)

    attention_output_a = multi_head_attention_layer(
        cdr3a_cat, pep_cat, pep_cat, embed_dim=128, num_heads=8)

   
    attention_output_b = multi_head_attention_layer(
        cdr3b_cat, pep_cat, pep_cat, embed_dim=128, num_heads=8)
  
    print(f"attention_output_a shape: {attention_output_a.shape}")
    print(f"attention_output_b shape: {attention_output_b.shape}")

    attention_output_pep_a = multi_head_attention_layer(
        pep_cat, cdr3a_cat, cdr3a_cat,
        embed_dim=128, num_heads=8)

    attention_output_pep_b = multi_head_attention_layer(
        pep_cat, cdr3b_cat, cdr3b_cat,
        embed_dim=128, num_heads=8)
  
    combined_attention_a = concatenate([attention_output_a, self_attention_cdr3a], axis=-1)
    combined_attention_b = concatenate([attention_output_b, self_attention_cdr3b], axis=-1)
    combined_attention_ab = concatenate([combined_attention_a, combined_attention_b], axis=-1)
    print(" combined_attention_ab shape:", combined_attention_ab.shape)

    combined_attention_pepa = concatenate([attention_output_pep_a, self_attention_pep_cat],
                                          axis=-1)
    combined_attention_pepb = concatenate([attention_output_pep_b, self_attention_pep_cat],
                                          axis=-1)
    print(" combined_attention_pepa shape:", combined_attention_pepa.shape)
    print(" combined_attention_pepb shape:", combined_attention_pepb.shape)
    combined_attention_pepa = Dense(80, activation='relu')(combined_attention_pepa)
    print("combined_attention_pepa shape:", combined_attention_pepa.shape)
    pepa_features = GlobalMaxPooling1D(name='pepa_features')(
        combined_attention_pepa)
    print(" pepa_features shape:", pepa_features.shape)

    combined_attention_pepb = Dense(80, activation='relu')(combined_attention_pepb)
    print("combined_attention_pepb shape:", combined_attention_pepb.shape)
    pepb_features = GlobalMaxPooling1D(name='pepb_features')(
        combined_attention_pepb)
    print("pepb_features shape:", pepb_features.shape)

    combined_attention_ab = Dense(80, activation='relu')(combined_attention_ab) 
    print(" combined_attention_ab shape:", combined_attention_ab.shape)
    tcr_features = GlobalMaxPooling1D(name='tcr_features')(
        combined_attention_ab)
    print("tcr_features shape:", tcr_features.shape)

    final_features = concatenate([tcr_features, pepa_features, pepb_features], axis=-1, name='final_features')
    print("final_features shape:", final_features.shape)
    dense = Dense(32, activation='relu')(final_features)
    dropout = Dropout(0.4)(dense)
    # dense = Dense(32)(final_features)
    # dense = Dense(32, activation='relu')(final_features) 
    out = Dense(1, activation='sigmoid')(dropout)

    model = Model(inputs=[cdra_in, cdrb_in, pep_in], outputs=[out])
    return model
# Call and compile the model
model = CNN_extra()
model.compile(loss="binary_crossentropy",
                optimizer=adam_v2.Adam(learning_rate=lr),
                metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(curve="ROC", name="ROCAUC")])

# Train model
print('Training..')
if not os.path.exists(root_model):
    os.makedirs(root_model)
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)
early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=0, mode='min', restore_best_weights=True)
checkpointer = keras.callbacks.ModelCheckpoint(os.path.join(save_model_path, 'CNN_extra_feature_epoch{epoch:03d}_AUC{val_ROCAUC:.6f}.hdf5'), save_weights_only=True)
# history = model.fit([tcrb_train, pep_train], y_train, validation_data=([tcrb_test, pep_test], y_test), epochs=epochs,
#                   batch_size=batchsize, verbose=1, callbacks=[early_stop, checkpointer])
train_inputs = [tcrB_train,tcra_train,pep_train]
validation_inputs = [tcrB_test,tcra_test, pep_test]
history = model.fit(train_inputs, y_train,
                    validation_data=(validation_inputs, y_test),
                    epochs=epochs, batch_size=batchsize, verbose=1, callbacks=[early_stop, checkpointer])


# Logs of results
history_dict = history.history 
epoch = []  
loss = history_dict['loss'] 
acc = history_dict['binary_accuracy']
auc_roc = history_dict['ROCAUC']
val_loss = history_dict['val_loss']
val_acc = history_dict['val_binary_accuracy']
val_auc_roc = history_dict['val_ROCAUC']
for ep in range(len(acc)):  
    epoch.append(ep + 1) 
dfhistory = {'epoch': epoch,
             'loss': loss, 'acc': acc, 'auc_roc': auc_roc,
             'val_loss': val_loss, 'val_acc': val_acc,
             'val_auc_roc': val_auc_roc}  
df = pd.DataFrame(
    dfhistory) 
if not os.path.exists(
        root_history):  # os.path.exists()
    os.makedirs(root_history)
df.to_csv(os.path.join(root_history, "CNN_extra_feature_{}.tsv".format(model_dir2)), header=True, sep='\t',
          index=False)  

# get num_epoch
val_auc_roc = list(df['val_auc_roc']) 
max_auc_index = val_auc_roc.index(max(val_auc_roc))  
num_epoch = max_auc_index + 1  
print("max auc epoch:", num_epoch)

val_loss = list(df['val_loss'])
min_val_loss_index = val_loss.index(min(val_loss))  
loss_num_epoch = min_val_loss_index + 1  
print("loss auc epoch:", loss_num_epoch)
# extra CNN feature
for root, dirs, files in os.walk(save_model_path):
    for file in files:
        file_path = os.path.join(save_model_path, file)
        if file.startswith("CNN_extra_feature_epoch{:03d}".format(num_epoch)):
            PATH = file_path
            model.load_weights(PATH,by_name=True)
          
            cat_cdr3_model_attention = Model(inputs=model.input,
                                             outputs=model.get_layer('tcr_features').output)
           
            cat_peptidecdr3a_model_attention = Model(inputs=model.input,
                                             outputs=model.get_layer('pepa_features').output)
            cat_peptidecdr3b_model_attention = Model(inputs=model.input,
                                             outputs=model.get_layer('pepb_features').output)
         
            train_inputs_full = [tcrB_train, tcra_train, pep_train]
            test_inputs_full = [tcrB_test, tcra_test, pep_test]
            
            cat_cdr3_attention_output_train = cat_cdr3_model_attention.predict(train_inputs_full)
            cat_peptidecdr3a_attention_output_train = cat_peptidecdr3a_model_attention.predict(train_inputs_full)
            cat_peptidecdr3b_attention_output_train = cat_peptidecdr3b_model_attention.predict(train_inputs_full)

            cat_cdr3_attention_output_test = cat_cdr3_model_attention.predict(test_inputs_full)
            cat_peptidecdr3a_attention_output_test = cat_peptidecdr3a_model_attention.predict(test_inputs_full)
            cat_peptidecdr3b_attention_output_test = cat_peptidecdr3b_model_attention.predict(test_inputs_full)


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
            os.remove(file_path)# # extra CNN feature

cdr3b_dict_train = {}
for i in range(len(train_data.A3)):
    if train_data.A3[i] + train_data.B3[i] not in cdr3b_dict_train.keys():
        cdr3b_dict_train[train_data.A3[i] + train_data.B3[i]] =cat_cdr3_attention_output_train[i]

peptide_cdr3a_dict_train = {}
for i in range(len(train_data.peptide)):
    if train_data.A3[i] + train_data.peptide[i] not in peptide_cdr3a_dict_train.keys():
        peptide_cdr3a_dict_train[train_data.A3[i] + train_data.peptide[i]] =  cat_peptidecdr3a_attention_output_train[i]

peptide_cdr3b_dict_train = {}
for i in range(len(train_data.peptide)):
    if train_data.B3[i]+train_data.peptide[i] not in peptide_cdr3b_dict_train.keys():
        peptide_cdr3b_dict_train[train_data.B3[i]+train_data.peptide[i]] =  cat_peptidecdr3b_attention_output_train[i]

cdr3b_dict_test = {}
for i in range(len(test_data.A3)):
    if test_data.A3[i] + test_data.B3[i] not in cdr3b_dict_test.keys():
        cdr3b_dict_test[test_data.A3[i] + test_data.B3[i]] = cat_cdr3_attention_output_test[i]

peptide_cdr3a_dict_test = {}
for i in range(len(test_data.peptide)):
    if test_data.A3[i] + test_data.peptide[i] not in peptide_cdr3a_dict_test.keys():
        peptide_cdr3a_dict_test[test_data.A3[i] + test_data.peptide[i]] = cat_peptidecdr3a_attention_output_test[i]

peptide_cdr3b_dict_test = {}
for i in range(len(test_data.peptide)):
    if test_data.B3[i]+test_data.peptide[i] not in peptide_cdr3b_dict_test.keys():
        peptide_cdr3b_dict_test[test_data.B3[i]+test_data.peptide[i]] = cat_peptidecdr3b_attention_output_test[i]

with open(train_path_pickle_cdr3, 'wb') as f1:
    pickle.dump(cdr3b_dict_train, f1)
with open(train_path_pickle_peptidecdr3a, 'wb') as f2:
    pickle.dump(peptide_cdr3a_dict_train, f2)
with open(train_path_pickle_peptidecdr3b, 'wb') as f3:
    pickle.dump(peptide_cdr3b_dict_train, f3)
print("train dataset CNN feature has saved!")

with open(test_path_pickle_cdr3, 'wb') as f4:
    pickle.dump(cdr3b_dict_test, f4)
with open(test_path_pickle_peptidecdr3a, 'wb') as f5:
    pickle.dump(peptide_cdr3a_dict_test, f5)
with open(test_path_pickle_peptidecdr3b, 'wb') as f6:
    pickle.dump(peptide_cdr3b_dict_test, f6)
print("test dataset CNN feature has saved!")
