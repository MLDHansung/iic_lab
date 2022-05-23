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
import math



 # to make the length of all MFCC equal
# mypath = "/home/iichsk/workspace/dataset/iic_respiratory/filtered_wav_dwt/"
#mypath = "/home/iichsk/workspace/dataset/iic_respiratory/wavfile/"
mypath = "/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files2"

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]
filepaths = [join(mypath, f) for f in filenames] # full paths of files


min_level_db= -100
def normalize_mel(S):
    return np.clip((S-min_level_db)/-min_level_db,0,1)


def extract_features_melspectrogram(file_name):
    y = librosa.load(file_name,16000)[0]
    S =  librosa.feature.melspectrogram(y=y, n_mels=80, n_fft=512, win_length=400, hop_length=160) # 320/80
    norm_log_S = normalize_mel(librosa.power_to_db(S, ref=np.max))
    return norm_log_S
                        



def extract_features(file_name):
    max_pad_len = 862
    max_pad_len_audio = 356394
    """
    This function takes in the path for an audio file as a string, loads it, and returns the MFCC
    of the audio"""

    try:
        
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=20)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        #print('audio=',audio)

        pad_width = max_pad_len - mfccs.shape[1]
        # pad_width_audio = max_pad_len_audio - audio.shape[0]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        #audio = np.pad(audio, pad_width=((0, 0), (math.trunc(pad_width_audio/2),math.floor(pad_width_audio/2))), mode='constant')
        times = np.arange(len(audio))/float(sample_rate)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccs,audio,times

spt_features = []
img_features = []


def extract_features_scalogram(file_name):
    sr=22050
    max_pad_len = 431
    """
    This function takes in the path for an audio file as a string, loads it, and returns the MFCC
    of the audio"""
   
    try:
        clip, sample_rate = librosa.load(file_name, sr=sr, mono=True) 
        #print('clip',clip.shape)
        clip_cqt = librosa.cqt(clip, hop_length=256, sr=sample_rate, fmin=30, bins_per_octave=32, n_bins=150, filter_scale=1.)    
        clip_cqt_abs = np.abs(clip_cqt)
        pad_width = max_pad_len - clip_cqt_abs.shape[1]
        # clip_cqt_abs = np.pad(clip_cqt_abs, pad_width=((0, 0), (math.trunc(pad_width/2),math.floor(pad_width/2))), mode='constant')
        clip_cqt_abs = np.pad(clip_cqt_abs, pad_width=((0, 0), (0,pad_width)), mode='constant')
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


waveform_len=[]
for file_name in filepaths:
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=20)
    times = np.arange(len(audio))/float(sample_rate)
    waveform = audio
    waveform_len.append(times.shape[0])
    #print(times.shape[0])
    fig = plt.figure(figsize=(8.62,2.40))
    plt.axis([0,20,-1,1])
    plt.plot(times,waveform, linewidth=0.5, color='b')
    plt.grid(True)
    # plt.xlim([0,10])
    plt.savefig('{}_waveform.png'.format(file_name))
    plt.close(fig)
print('max times=',max(waveform_len))
#########################save scalogram to png file###########################

# for file_name in filepaths:
#     data, _, time = extract_features_scalogram(file_name)
#     fig = plt.figure(figsize=(8.62,2.40))
#     plt.figure(1)
#     librosa.display.specshow(data, y_axis='cqt_hz',x_axis='time')
#     plt.colorbar(format='%+2.0f dB')
#     plt.tight_layout()
#     plt.savefig('{}_scalo.png'.format(file_name))
#     plt.close()

#########################save MFCC to png file###########################

# cnt = 0
# for file_name in filepaths:
#     cnt += 1
#     print('save mfcc image number {}...'.format(cnt))
#     data, _, time = extract_features(file_name)
#     #data = feature_extraction_melspectrogram(file_name)

#     plt.figure(1)
#     librosa.display.specshow(data, y_axis='mel',x_axis='time')
#     plt.colorbar(format='%+2.0f dB')
#     plt.tight_layout()
#     plt.savefig('{}_mfcc.png'.format(file_name))
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