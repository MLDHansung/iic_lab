from datetime import datetime
from os import listdir
from os.path import isfile, join
from PIL import Image

import librosa
import librosa.display
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
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

batch_size = 128
learning_rate = 1e-3
filter_size1 = 5
filter_size2 = 3
filter_size3 = (2,10)
filter_size4 = (2, 10)
classes = 4
spt_dropout = 0.3
scalo_dropout = 0.3
img_dropout = 0.3

mypath = "/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/processed_audio_files_nopad"
mypath_waveform = "/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/waveform_image"

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]
filenames_waveform = [f for f in listdir(mypath_waveform) if (isfile(join(mypath_waveform, f)) and f.endswith('.png'))]

p_id_in_file = [] # patient IDs corresponding to each file
p_id_in_file2 = [] # patient IDs corresponding to each file
p_id_in_file_waveform = [] # patient IDs corresponding to each file

for name in filenames:
    p_id_in_file.append(int(name.split('_')[0]))
p_id_in_file = np.array(p_id_in_file)
print('wave=',p_id_in_file)
for name in filenames:
    p_id_in_file2.append(int(name.split('_')[0]))
p_id_in_file2 = np.array(p_id_in_file2)
print('wave=',p_id_in_file2)
for name in filenames:
    p_id_in_file_waveform.append(int(name.split('_')[0]))
p_id_in_file_waveform = np.array(p_id_in_file_waveform)
print('wave=',p_id_in_file_waveform)
def extract_features(file_name):
    """
    This function takes in the path for an audio file as a string, loads it, and returns the MFCC
    of the audio"""
    max_pad_len = 862
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=20)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        #mfccs = np.pad(mfccs, pad_width=((0,0)), mode='constant')
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccs

sr=22050

def extract_features_scalogram(file_name):
    """
    This function takes in the path for an audio file as a string, loads it, and returns the MFCC
    of the audio"""
    max_pad_len = 1393
    try:
        clip, sample_rate = librosa.load(file_name, sr=sr, mono=True) 
        #print('clip',clip.shape)
        clip_cqt = librosa.cqt(clip, hop_length=256, sr=sample_rate, fmin=30, bins_per_octave=32, n_bins=150, filter_scale=1.)    
        clip_cqt_abs = np.abs(clip_cqt)
        pad_width = max_pad_len - clip_cqt_abs.shape[1]
        clip_cqt_abs = np.pad(clip_cqt_abs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        times = np.arange(len(clip))/float(sample_rate)
        
        #clip_cqt_abs=np.log(clip_cqt_abs**2)
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return clip_cqt_abs, clip, times

def extract_image(file_name):
    #file_name=
    im = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    #im = plt.resize(im, (640, 480))
    #print('file_name=',file_name)
    return im

filepaths = [join(mypath, f) for f in filenames] # full paths of files
filepaths_waveform = [join(mypath_waveform, f)+'_waveform.png' for f in filenames] # full paths of files

p_diag = pd.read_csv("/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/csv_data/data2.csv",header=None)

labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])
labels2 = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file2])
labels_waveform = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file_waveform])

spt_features = []
scalo_features = []
img_features = []


for file_name in filepaths:
    data = extract_features(file_name)
    #print("file_name=",file_name)
    spt_features.append(data)

for file_name1 in filepaths:
    data,_,_ = extract_features_scalogram(file_name1)
    #print("file_name=",file_name1)
    scalo_features.append(data)

for file_name2 in filepaths_waveform:
    data = extract_image(file_name2)
    #print("file_name=",file_name2)
    img_features.append(data)

print('Finished feature extraction from ', len(spt_features), ' files')
print('Finished feature extraction from ', len(scalo_features), ' files')
print('Finished feature extraction from ', len(img_features), ' files')

spt_features = np.array(spt_features)
spt_features1 = np.delete(spt_features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)
labels1 = np.delete(labels, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)

scalo_features = np.array(scalo_features)
scalo_features1 = np.delete(scalo_features, np.where((labels2 == 'Asthma') | (labels2 == 'LRTI'))[0], axis=0)
scalo_labels1 = np.delete(labels2, np.where((labels2 == 'Asthma') | (labels2 == 'LRTI'))[0], axis=0)

img_features = np.array(img_features)
img_features1 = np.delete(img_features, np.where((labels_waveform == 'Asthma') | (labels_waveform == 'LRTI'))[0], axis=0)
waveform_labels1 = np.delete(labels_waveform, np.where((labels_waveform == 'Asthma') | (labels_waveform == 'LRTI'))[0], axis=0)

unique_elements, counts_elements = np.unique(labels1, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

le = LabelEncoder()
i_labels = le.fit_transform(labels1)
i_labels2 = le.fit_transform(scalo_labels1)
i_labels3 = le.fit_transform(waveform_labels1)

oh_labels = to_categorical(i_labels)
oh_labels2 = to_categorical(i_labels2)
oh_labels3 = to_categorical(i_labels3)

spt_features1 = np.reshape(spt_features1, (*spt_features1.shape, 1))
scalo_features1 = np.reshape(scalo_features1, (*scalo_features1.shape, 1))
img_features1 = np.reshape(img_features1, (*img_features1.shape, 1))

spt_train, spt_test, label_train, label_test = train_test_split(spt_features1, oh_labels, stratify=i_labels,
                                                    test_size=0.2, random_state = 42)
scalo_train, scalo_test, label2_train, label2_test = train_test_split(scalo_features1, oh_labels2, stratify=i_labels2,
                                                    test_size=0.2, random_state = 42)
waveform_train, waveform_test, label3_train, label3_test = train_test_split(img_features1, oh_labels3, stratify=i_labels3,
                                                    test_size=0.2, random_state = 42)
print(spt_train.shape)
# print('spt=',spt_train[0:2,0,0,0])
# print('scalo=',scalo_train[0:2,0,0,0])
# print('wave=',waveform_train[0,:,:,0])
# print('wave=',waveform_test[0,:,:,0])
df = pd.DataFrame(waveform_test[0,:,:,0])
df.to_excel("/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/wave_value_.xlsx", sheet_name = 'sheet1')

spt_train = np.transpose(spt_train, (0,3,1,2))
spt_test = np.transpose(spt_test, (0,3,1,2))
scalo_train = np.transpose(scalo_train, (0,3,1,2))
scalo_test = np.transpose(scalo_test, (0,3,1,2))
waveform_train = np.transpose(waveform_train, (0,3,1,2))
waveform_test = np.transpose(waveform_test, (0,3,1,2))

#data loader