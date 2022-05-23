import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
import soundfile as sf
import keras
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras import regularizers
from keras.layers import PReLU
import cv2
import librosa
inputData = np.empty((6898,110250))
targetData = np.empty(6898)

root = '/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/'
filenames = [s.split('.')[0] for s in os.listdir(path = root) if '.txt' in s]
i_list = []
rec_annotations = []
rec_annotations_dict = {}
inputImageData = np.empty((6898,32,216))
def Extract_Annotation_Data(file_name, root):
    tokens = file_name.split('_')
    recording_info = pd.DataFrame(data = [tokens], columns = ['Patient number', 'Recording index', 'Chest location','Acquisition mode','Recording equipment'])
    recording_annotations = pd.read_csv(os.path.join(root, file_name + '.txt'), names = ['Start', 'End', 'Crackles', 'Wheezes'], delimiter= '\t')
    return (recording_info, recording_annotations)


def getPureSample(start,end,raw_data, sr=22050):
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

def getClass(df,index):
    if(df.at[index,'Wheezes']==0 and df.at[index,'Crackles']==0):
        return 0
    elif(df.at[index,'Wheezes']==1 and df.at[index,'Crackles']==0):
        return 1
    elif(df.at[index,'Wheezes']==0 and df.at[index,'Crackles']==1):
        return 2
    elif(df.at[index,'Wheezes']==1 and df.at[index,'Crackles']==1):
        return 3
for s in filenames:
    (i,a) = Extract_Annotation_Data(s, root)
    i_list.append(i)
    rec_annotations.append(a)
    rec_annotations_dict[s] = a
recording_info = pd.concat(i_list, axis = 0)
del i_list
del rec_annotations
l=0
for i in rec_annotations_dict:
    j = rec_annotations_dict[i]
    for k in range(j.shape[0]):
        data,sampleRate = librosa.load(root+i+'.wav')
        reqLen=6*sampleRate
        data = getPureSample(j.at[k,'Start'],j.at[k,'End'], data, sampleRate)

        #data = slice_data(j.at[k,'Start'],j.at[k,'End'], data, sampleRate)

        targetData[l] = getClass(j,k)
        save_path = '/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/slice_wav_data/' + str(l) + '.wav'
        sf.write(file=save_path,data=data,samplerate=sampleRate)
        print('saving data no.',l,'....')
        l=l+1
del rec_annotations_dict
targetData.to_csv('/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/csv_data/targetdata.csv')
print('targetData',targetData.shape)
