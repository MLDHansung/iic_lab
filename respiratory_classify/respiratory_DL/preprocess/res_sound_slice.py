import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import os
import soundfile as sf

patient_data=pd.read_csv('/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis4.csv',names=['pid','disease'])
path='/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/'
files=[s.split('.')[0] for s in os.listdir(path) if '.txt' in s]
print('files',files)

def getPureSample(raw_data,start,end,sr=22050):
    '''
    Takes a numpy array and spilts its using start and end args
    
    raw_data=numpy array of audio sample
    start=time
    end=time
    sr=sampling_rate
    mode=mono/stereo
    
    '''
    max_ind = len(raw_data) 
    start_ind = min(int(start * sr), max_ind)
    end_ind = min(int(end * sr), max_ind)
    return raw_data[start_ind: end_ind]





def getFilenameInfo(file):
    return file.split('_')

def slicing_data(file_name):
    files_data=[]
    for file in files:
        if file == file_name:
            data=pd.read_csv(path + file + '.txt',sep='\t',names=['start','end','crackles','wheezes'])
            name_data=getFilenameInfo(file)
            data['pid']=name_data[0]
            data['mode']=name_data[-2]
            data['filename']=file
            files_data.append(data)
    files_df=pd.concat(files_data)
    files_df.reset_index()


    patient_data.pid=patient_data.pid.astype('int32')
    files_df.pid=files_df.pid.astype('int32')

    data=pd.merge(files_df,patient_data,on='pid')
    print(data.head())

    os.makedirs('csv_data', exist_ok=True)
    data.to_csv('/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/csv_data/data_nopad.csv',index=False)

    i,c=0,0
    cnt = 0
    for index,row in data.iterrows():
        maxLen=6
        start=row['start']
        end=row['end']
        filename=row['filename']
        pid=row['pid']
        crackles=row['crackles']
        wheezes=row['wheezes']
        #If len > maxLen , change it to maxLen
        #print('start: ',start,' end :', end)
        
        audio_file_loc=path + filename + '.wav'
        
        if index > 0:
            #check if more cycles exits for same patient if so then add i to change filename
            if data.iloc[index-1]['filename']==filename:
                i+=1
            else:
                i=0
        filename= str(c) + '_' + filename + '.wav'
        os.makedirs('/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/processed_audio_files_nopad/', exist_ok=True)

        save_path='/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/processed_audio_files_nopad/' + filename
        c+=1
        audioArr,sampleRate=librosa.load(audio_file_loc)
        pureSample=getPureSample(audioArr,start,end,sampleRate)
        pureSample_list = np.append(pureSample)
        print('pureSample_list',pureSample_list.shape)
        cnt+=1
        if cnt%100==0:
            print('slicing number ', cnt)
        #pad audio if pureSample len < max_len
        #reqLen=6*sampleRate
        #padded_data = librosa.util.pad_center(pureSample, reqLen)
        
        # sf.write(file=save_path,data=pureSample,samplerate=sampleRate)
    return pureSample_list
