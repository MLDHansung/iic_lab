from datetime import datetime
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import to_categorical
import numpy as np
import pandas as pd

import time


# dataset load
mypath = "/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/processed_audio_files_nopad/" # respiratory sound directory
p_diag = pd.read_csv("disease_result.csv",header=None) # patient diagnosis csv file
p_id_testset = pd.read_csv("p_id_testset2.csv",header=None) # patient diagnosis csv file

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]

p_id_in_file = [] # patient IDs corresponding to each file

cnt2 = 0
for s_name in p_id_testset[0]:
    cnt = 0
    for d_name in p_diag[0]:
        if(s_name.split('_')[1]+s_name.split('_')[2]+s_name.split('_')[3]+s_name.split('_')[4] == d_name.split('_')[0]+d_name.split('_')[1]+d_name.split('_')[2]+d_name.split('_')[3]):
            p_id_testset.iloc[cnt2,1]='{}_{}'.format(p_diag.iloc[cnt,1],p_id_testset.iloc[cnt2,1])
            p_id_testset.iloc[cnt2,2]='{}_{}'.format(p_diag.iloc[cnt,1],p_id_testset.iloc[cnt2,2])
        cnt+=1
    cnt2+=1
print(p_id_testset)
p_id_testset=np.array(p_id_testset)
p_id_testset=pd.DataFrame(p_id_testset,columns=['file name','prediction','target'])
p_id_testset.to_csv("matching_result_testset.csv",index=False)



