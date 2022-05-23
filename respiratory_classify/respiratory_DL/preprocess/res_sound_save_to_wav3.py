import matplotlib.pyplot as plt

from datetime import datetime
from os import listdir
from os.path import isfile, join
from PIL import Image

import librosa
import librosa.display
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import cv2
from torch.autograd import Variable
import time
import cwt
from scipy import interpolate

max_pad_len = 259 # to make the length of all MFCC equal
mypath = "/home/iichsk/workspace/dataset/iic_respiratory/wavfile/"

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]
filepaths = [join(mypath, f) for f in filenames] # full paths of files


min_level_db= -100
def normalize_mel(S):
    return np.clip((S-min_level_db)/-min_level_db,0,1)


def feature_extraction_melspectrogram(file_name):
    y = librosa.load(file_name,16000)[0]
    S =  librosa.feature.melspectrogram(y=y, n_mels=80, n_fft=512, win_length=400, hop_length=160) # 320/80
    norm_log_S = normalize_mel(librosa.power_to_db(S, ref=np.max))
    return norm_log_S
                        



def extract_features(file_name):
    """
    This function takes in the path for an audio file as a string, loads it, and returns the MFCC
    of the audio"""

    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=6)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        
        pad_width = max_pad_len - mfccs.shape[1]
        #mfccs = np.pad(mfccs, pad_width=((0, 0), (0,pad_width)), mode='constant')
        #mfccs = np.pad(mfccs, pad_width=((0,0)), mode='constant')
        #f = interpolate.interp2d(x, y, mfccs, kind='linear')
        #x_new = np.arange(0, 697)
        #y_new = np.arange(0, 40)
        #mfccs = f(x_new, y_new)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccs,audio,sample_rate

spt_features = []
img_features = []

sr=22050
def extract_features_scalogram(file_name):
    """
    This function takes in the path for an audio file as a string, loads it, and returns the MFCC
    of the audio"""
   
    try:
        clip, sample_rate = librosa.load(file_name, sr=sr, mono=True) 
        print('clip',clip.shape)
        clip_cqt = librosa.cqt(clip, hop_length=128, sr=sample_rate, fmin=30, bins_per_octave=32, n_bins=250, filter_scale=1.)
        clip_cqt_abs = np.abs(clip_cqt)
        times = np.arange(len(clip))/float(sample_rate)
        #clip_cqt_abs=np.log(clip_cqt_abs**2)
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return clip_cqt_abs, clip, times

def extract_image(file_name):
    im = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    #print(im.shape)
    return im
# Iterate through each sound file and extract the features
#########################save wav to png file###########################
cnt = 0
waveform_list = []
# for file_name in filepaths:
#     if(cnt % 1000 == 0):
#         print('save waveform image number {}...'.format(cnt))
#     data, waveform,sample_rate = extract_features(file_name)
#     time =np.linspace(0, len(waveform)/sample_rate,len(waveform))
#     #print(waveform.shape)
#     #waveform_list.append([file_name,time[-1:]])
#     fig = plt.figure(figsize=(6.40,2.40))
#     plt.plot(time, waveform, linewidth=0.5, color='b')
#     plt.axis()
#     plt.xlim([0,6])
#     plt.grid(True)
#     plt.savefig('{}_waveform.png'.format(file_name))
#     plt.close(fig)
#     cnt+=1

for file_name in filepaths:
    if(cnt % 1000 == 0):
        print('save waveform image number {}...'.format(cnt))
    data, waveform,sample_rate = extract_features(file_name)
    time =np.linspace(0, len(waveform)/sample_rate,len(waveform))
    #print(waveform.shape)
    #waveform_list.append([file_name,time[-1:]])
    fig = plt.figure(figsize=(6.40,2.40))
    plt.plot(time, waveform, linewidth=0.5, color='b')
    # plt.ylim([0,100])
    librosa.display.specshow(data)
    plt.colorbar(format='%+2.0f dB')
    plt.axis()
    # plt.xlim([0,6])
    #plt.grid(True)
    plt.savefig('{}_MFCC.png'.format(file_name))
    plt.close(fig)
    cnt+=1
# waveform_np=np.array(waveform_list)
# print(waveform_np.shape)
# df2=pd.DataFrame.from_records(waveform_np[:,:])
# df2.to_excel('/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/processed_audio_files_nopad/_waveform_time.xlsx')
# waveform_np_time=np.array(waveform_np[:,1])
# print(waveform_np_time)
# waveform_min=np.argmin(waveform_np[:,1:2])
# print('waveform_min',waveform_min)
# min_idex=np.where(waveform_np_time<1)
# print('min_index',min_index)

# for file_name in filepaths:
#     cnt += 1
#     if(cnt % 1000 == 0):
#         print('save mfcc image number {}...'.format(cnt))
#     #mfcc_data, _,_ = extract_features(file_name)
#     mel_data = feature_extraction_melspectrogram(file_name)
#     #scal_data,_,_ = extract_features_scalogram(file_name)

#     plt.figure(1)
#     librosa.display.specshow(mel_data,x_axis='time')

#     plt.colorbar(format='%+2.0f dB')
#     #f, (ax1) = plt.subplots(1, 2, sharey=True, figsize=(15, 8))
#     #ax1.imshow(scal_data, origin='lower', aspect=2.)
#     plt.tight_layout()
#     plt.savefig('{}mel_data.png'.format(file_name))
#     plt.close()

    ###########################################################################


'''for i in range(0,2):
    for ii in range(0,2):
        mypath = "/home/iichsk/workspace/dataset/facedatabase/wav_sym/training/{}_{}".format(i,ii)

        filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]
        filepaths = [join(mypath, f) for f in filenames] # full paths of files


        spt_features = []
        img_features = []

        def extract_image(file_name):
            im = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            #print(im.shape)
            return im
        # Iterate through each sound file and extract the features
        #########################save wav to png file###########################
        for file_name in filepaths:
            data, waveform = extract_features(file_name)

            fig = plt.figure(figsize=(8.62,2.40))
            plt.axis([0,22,-35000,35000])
            plt.plot(waveform, linewidth=0.5, color='b')

            plt.grid(True)
            plt.savefig('{}_temp.png'.format(file_name))
            plt.close(fig)
     '''