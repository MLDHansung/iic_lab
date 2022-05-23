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
from matplotlib import cm
from sklearn.manifold import TSNE
from tsne import bh_sne

# dataset load
mypath = "/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/processed_audio_files_nopad/" # respiratory sound directory
p_diag = pd.read_csv("/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/csv_data/symptom_label.csv",header=None) # patient diagnosis csv file

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]

p_id_in_file = [] # patient IDs corresponding to each file
cnt = 0
cnt2 = 0
for x in p_diag[0]:
    for name in filenames:
        if(int(x) == int(name.split('_')[0])):
            if(p_diag.iloc[cnt2,1] == 'normal' and p_diag.iloc[cnt2,9] != 'Asthma' and p_diag.iloc[cnt2,9] != 'LRTI'):
                cnt+=1
                p_diag.iloc[cnt2,1]='{}_{}'.format(p_diag.iloc[cnt2,1],p_diag.iloc[cnt2,9])
                p_id_in_file.append(int(name.split('_')[0]))
    cnt2+=1
p_id_in_file = np.array(p_id_in_file)
dataset_p_id = p_id_in_file
p_diag.to_csv("/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/csv_data/symptom_label_wheeze_d.csv")

labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in dataset_p_id])

dataset_p_id = np.delete(dataset_p_id, np.where((labels == 'Asthma') | (labels == 'LRTI') | (labels == 'd_normal_Asthma') | (labels == 'd_normal_LRTI'))[0], axis=0)
labels = np.delete(labels, np.where((labels == 'Asthma') | (labels == 'LRTI') | (labels == 'd_normal_Asthma') | (labels == 'd_normal_LRTI'))[0], axis=0)

print('Finished feature extraction from ', len(dataset_p_id), ' files')
unique_elements, counts_elements = np.unique(labels, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

le = LabelEncoder()
i_labels = le.fit_transform(labels)
oh_labels = to_categorical(i_labels)
test_label = oh_labels

def extract_MFCC(file_name):
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
     
    return clip_cqt_abs


mfcc_testset=[]
for p_id in dataset_p_id:
    for file_name in listdir(mypath):
        if (int(file_name.split('_')[0]) == p_id):

            extract_mfcc_data = extract_MFCC((mypath+file_name))
            mfcc_testset.append(extract_mfcc_data)
 

mfcc_testset = np.array(mfcc_testset)
mfcc_testset = np.reshape(mfcc_testset, (*mfcc_testset.shape, 1))
mfcc_testset = np.transpose(mfcc_testset, (0,3,1,2))
np.save('mfcc_normal_d.npy',mfcc_testset)
np.save('dataset_label_normal_d.npy',test_label)