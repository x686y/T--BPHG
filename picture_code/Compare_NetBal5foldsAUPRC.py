import matplotlib.pyplot as plt
import numpy as np

# 
models = ['TITAN', 'epiTCR', 'TEINet', 'MixTCRpred', 'NetTCR-2.2',
          'TSpred_CNN', 'TSpred_att', 'TSpred_ens', 'HeteroTCR', 'Tαβ-BPHG']


auprc_values = [0.347, 0.361, 0.336, 0.364, 0.349, 0.366, 0.397, 0.397, 0.393, 0.765]
std_devs = [0.011, 0.027, 0.014, 0.041, 0.027, 0.039, 0.067, 0.068, 0.013, 0.049]


mean_auprc = np.mean(auprc_values)  # AUPRC 
std_auprc = np.std(auprc_values)    

# 输出计算结果
print(f"Mean AUPRC: {mean_auprc:.3f}")
print(f"Standard Deviation of AUPRC: {std_auprc:.3f}")


colors = [(254/255,254/255,215/255),  
          (238/255,248/255,180/255), 
          (205/255,235/255,179/255),  
          (149/255,213/255,184/255), 
          (91/255,191/255,192/255), 
          (48/255,165/255,194/255),  
          (30/255,128/255,184/255),  
          (34/255,84/255,163/255), 
          (33/255,49/255,140/255), 
          (8/255,29/255,89/255)] 

#
plt.figure(figsize=(12, 8))


bars = plt.bar(models, auprc_values, color=colors)


plt.title('NetTCR_bal',fontsize=18 ,fontweight='bold')
plt.xlabel('Model', fontsize=18 ,fontweight='bold')
plt.ylabel('AUPRC', fontsize=18 ,fontweight='bold')


# plt.text(-0.05, 1.05, 'B', fontsize=22, fontweight='bold', va='center', ha='center', transform=plt.gca().transAxes)


legend_labels = [f'{models[i]} (AUPRC={auprc_values[i]:.3f} ± {std_devs[i]:.3f})' for i in range(len(models))]


plt.legend(bars, legend_labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)


plt.xticks(rotation=45, ha='right')


plt.tick_params(axis='x', labelsize=18) 
plt.tick_params(axis='y', labelsize=18) 

ax = plt.gca()  
for spine in ax.spines.values():
    spine.set_edgecolor('black') 
    spine.set_linewidth(1) 


plt.tight_layout()


save_path = r'D:\xy\HeteroTCR-main\5folds_visual\picture\compareNetbalAUPRC.png'
plt.savefig(save_path, dpi=300)


plt.show()
