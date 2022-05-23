
import pandas as pd



# dataset load
p_diag = pd.read_csv("/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/csv_data/symptom_label_5classes.csv",header=None) # patient diagnosis csv file


p_id_in_file = [] # patient IDs corresponding to each file
cnt = 0
cnt2 = 0
for x in p_diag[0]:
    if(p_diag.iloc[cnt2,1] == 'normal' and p_diag.iloc[cnt2,9] != 'zHealthy'):
        p_diag.iloc[cnt2,1]='d_{}'.format(p_diag.iloc[cnt2,1])
    cnt2+=1

p_diag.to_csv("/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/csv_data/symptom_label_5classes.csv",index=False)

