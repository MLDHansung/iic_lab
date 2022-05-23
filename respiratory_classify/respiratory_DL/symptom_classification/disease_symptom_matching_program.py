from datetime import datetime
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import time

symptom_result = pd.read_csv("/home/iichsk/workspace/result/symptom_result.csv",header=None)
disease_result = pd.read_csv("/home/iichsk/workspace/result/disease_result.csv",header=None,encoding='cp949')
symptom_result = np.array(symptom_result)
disease_result = np.array(disease_result)


matching_list=[]
for i in range(int(disease_result.shape[0])):
    d_p_id = disease_result[i,0]
    d_p_id = d_p_id.split('_')[0]+'_'+d_p_id.split('_')[1]+'_'+d_p_id.split('_')[2]+'_'+d_p_id.split('_')[3]+'_'+d_p_id.split('_')[4]

    # print(d_p_id)
    for ii in range(int(symptom_result.shape[0])):
        s_p_id = symptom_result[ii,0]
        s_p_id = s_p_id.split('_')[1]+'_'+s_p_id.split('_')[2]+'_'+s_p_id.split('_')[3]+'_'+s_p_id.split('_')[4]+'_'+s_p_id.split('_')[5]
        if (d_p_id==s_p_id):
            matching_list.append(np.concatenate((disease_result[i,0:1],disease_result[i,1:2],disease_result[i,2:3],symptom_result[ii,0:1],symptom_result[ii,1:2],symptom_result[ii,2:3])))

matching_list= np.array(matching_list)
print(matching_list)
matching_list_df = pd.DataFrame(matching_list)
count_t = time.time()
matching_list_df.to_excel("/home/iichsk/workspace/result/matching_list{}.xlsx".format(count_t), sheet_name = 'sheet1')