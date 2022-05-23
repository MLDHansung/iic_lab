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
import torchvision

import cv2
from torch.autograd import Variable
import time

from torchviz import make_dot

batch_size = 128
learning_rate = 1e-3
filter_size1 = 10
filter_size2 = 2
classes = 6
spt_dropout = 0.325
img_dropout = 0.325

mypath = "/home/iichsk/workspace/respiratory_classify/database/disease_audio/audio_and_txt_files2/"
mypath2 = "/home/iichsk/workspace/respiratory_classify/database/disease_waveform_image/audio_and_txt_files2/"

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]
filenames2 = [f for f in listdir(mypath2) if (isfile(join(mypath2, f)) and f.endswith('.png'))]

p_id_in_file = [] # patient IDs corresponding to each file
p_id_in_file2 = [] # patient IDs corresponding to each file

for name in filenames:
    p_id_in_file.append(int(name[:3]))

p_id_in_file = np.array(p_id_in_file)

for name in filenames2:
    p_id_in_file2.append(int(name[:3]))

p_id_in_file2 = np.array(p_id_in_file2)
max_pad_len = 862 # to make the length of all MFCC equal

def extract_features(file_name):
    """
    This function takes in the path for an audio file as a string, loads it, and returns the MFCC
    of the audio"""

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

filepaths = [join(mypath, f) for f in filenames] # full paths of files
filepaths2 = [join(mypath2, f) for f in filenames2] # full paths of files

p_diag = pd.read_csv("/home/iichsk/workspace/respiratory_classify/database/csv_data/patient_diagnosis.csv",header=None)

labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])
labels2 = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file2])


spt_features = []
img_features = []
def extract_image(file_name):
    im = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
  
    return im

# Iterate through each sound file and extract the features
for file_name in filepaths:
    data = extract_features(file_name)
    spt_features.append(data)

for file_name in filepaths2:
    data = extract_image(file_name)
    img_features.append(data)

print('Finished feature extraction from ', len(spt_features), ' files')
print('Finished feature extraction from ', len(img_features), ' files')

spt_features = np.array(spt_features)
spt_features1 = np.delete(spt_features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)
labels1 = np.delete(labels, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)

img_features = np.array(img_features)
img_features1 = np.delete(img_features, np.where((labels2 == 'Asthma') | (labels2 == 'LRTI'))[0], axis=0)
img_labels1 = np.delete(labels2, np.where((labels2 == 'Asthma') | (labels2 == 'LRTI'))[0], axis=0)

unique_elements, counts_elements = np.unique(labels1, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))


le = LabelEncoder()
i_labels = le.fit_transform(labels1)
i_labels2 = le.fit_transform(img_labels1)

oh_labels = to_categorical(i_labels)
oh_labels2 = to_categorical(i_labels2)

spt_features1 = np.reshape(spt_features1, (*spt_features1.shape, 1))
img_features1 = np.reshape(img_features1, (*img_features1.shape, 1))

spt_train, spt_test, label_train, label_test = train_test_split(spt_features1, oh_labels, stratify=i_labels,
                                                    test_size=0.2, random_state = 42)
img_train, img_test, label2_train, label2_test = train_test_split(img_features1, oh_labels2, stratify=i_labels2,
                                                    test_size=0.2, random_state = 42)

spt_train = np.transpose(spt_train, (0,3,1,2))
spt_test = np.transpose(spt_test, (0,3,1,2))
img_train = np.transpose(img_train, (0,3,1,2))
img_test = np.transpose(img_test, (0,3,1,2))

# construct model on cuda if available
use_cuda = torch.cuda.is_available()

# loss
criterion = nn.MultiLabelSoftMarginLoss()

# FC layer 제외한 MFCC CNN 
class spt_cnn(nn.Module):

    def __init__(self):
        super(spt_cnn, self).__init__()

        self.spt_conv1 = nn.Conv2d(1, 16, kernel_size=2)  # 16@39*861
        self.spt_pool1 = nn.MaxPool2d(2)  # 16@19*430
        self.spt_dropout1 = nn.Dropout(spt_dropout)
        self.spt_conv2 = nn.Conv2d(16, 32, kernel_size=2)  # 32@18*429
        self.spt_pool2 = nn.MaxPool2d(2)  # 32@9*214
        self.spt_dropout2 = nn.Dropout(spt_dropout)
        self.spt_conv3 = nn.Conv2d(32, 64, kernel_size=2)  # 64@8*213
        self.spt_pool3 = nn.MaxPool2d(2)  # 64@4*106
        self.spt_dropout3 = nn.Dropout(spt_dropout)
        self.spt_conv4 = nn.Conv2d(64, 128, kernel_size=2)  # 128@3*105
        self.spt_pool4 = nn.MaxPool2d(2)  # 128@1*52
        self.spt_dropout4 = nn.Dropout(spt_dropout)
        self.spt_global_pool = nn.AdaptiveAvgPool2d(1)
        self.spt_fc1 = nn.Linear(128, 64)
        self.spt_bn1 = torch.nn.BatchNorm1d(64)
        self.spt_fc2 = nn.Linear(64, classes)  
        self.spt_relu = nn.ReLU()
    
    def forward(self, spt_x):

        spt_x = self.spt_relu(self.spt_conv1(spt_x))
        spt_x = self.spt_pool1(spt_x)  #
        spt_x = self.spt_dropout1(spt_x)
        spt_x = self.spt_relu(self.spt_conv2(spt_x))
        spt_x = self.spt_pool2(spt_x)  #
        spt_x = self.spt_dropout2(spt_x)
        spt_x = self.spt_relu(self.spt_conv3(spt_x))
        spt_x = self.spt_pool3(spt_x)  #
        spt_x = self.spt_dropout3(spt_x)  #
        spt_x = self.spt_relu(self.spt_conv4(spt_x))
        spt_x = self.spt_pool4(spt_x)  #
        spt_x = self.spt_dropout4(spt_x)  #
        spt_x = self.spt_global_pool(spt_x) # batchsize x netdim x 1 x 1
        spt_feature_x = spt_x
        spt_x = spt_x.view(spt_x.size(0), -1) # batchsize x netdim 
        spt_x = self.spt_fc1(spt_x) 
        spt_x = self.spt_bn1(spt_x)
        spt_x = self.spt_fc2(spt_x)

        return spt_x

spt_cnn = spt_cnn()
spt_cnn.cuda()
# MFCC CNN load 중
spt_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/disease_classify_best_weight/spt_cnn_weight_6_600_128_best.pt"))
spt_optimizer = optim.Adam(spt_cnn.parameters(), lr=learning_rate)

# FC layer 제외한 Waveform CNN 
class img_cnn(nn.Module):

    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(img_cnn, self).__init__()

        self.img_conv1 = nn.Conv2d(1, 16, kernel_size=filter_size1)  # 16@39*861
        self.img_pool1 = nn.MaxPool2d(2)  # 16@19*430
        self.img_dropout1 = nn.Dropout(img_dropout)
        self.img_conv2 = nn.Conv2d(16, 32, kernel_size=filter_size2)  # 32@18*429
        self.img_pool2 = nn.MaxPool2d(2)  # 32@9*214
        self.img_dropout2 = nn.Dropout(img_dropout)
        self.img_conv3 = nn.Conv2d(32, 64, kernel_size=filter_size2)  # 64@8*213
        self.img_pool3 = nn.MaxPool2d(2)  # 64@4*106
        self.img_dropout3 = nn.Dropout(img_dropout)
        self.img_conv4 = nn.Conv2d(64, 128, kernel_size=filter_size2)  # 128@3*105
        self.img_pool4 = nn.MaxPool2d(2)  # 128@1*52
        self.img_dropout4 = nn.Dropout(img_dropout)
        self.img_conv5 = nn.Conv2d(128, 256, kernel_size=filter_size2)  # 128@3*105
        self.img_pool5 = nn.MaxPool2d(2)  # 128@1*52
        self.img_dropout5 = nn.Dropout(img_dropout)
        self.img_global_pool = nn.AdaptiveAvgPool2d(1)
        self.img_fc1 = nn.Linear(256, 128)
        self.img_fc2 = nn.Linear(128, classes)
        
        self.img_relu = nn.ReLU()
       
    
    def forward(self, img_x):
        
        img_x = self.img_relu(self.img_conv1(img_x))
        img_x = self.img_pool1(img_x)  #
        img_x = self.img_dropout1(img_x)
        img_x = self.img_relu(self.img_conv2(img_x))
        img_x = self.img_pool2(img_x)  #
        img_x = self.img_dropout2(img_x)
        img_x = self.img_relu(self.img_conv3(img_x))
        img_x = self.img_pool3(img_x)  #
        img_x = self.img_dropout3(img_x)  #
        img_x = self.img_relu(self.img_conv4(img_x))
        img_x = self.img_pool4(img_x)  #
        img_x = self.img_dropout4(img_x)  #
        img_x = self.img_relu(self.img_conv5(img_x))
        img_x = self.img_pool5(img_x)  #
        img_x = self.img_dropout5(img_x)  
        img_x = self.img_global_pool(img_x)
        img_feature_x = img_x
        img_x = img_x.view(img_x.size(0), -1)
        img_x = self.img_fc1(img_x)  #
        img_x = self.img_fc2(img_x)  #

        return img_x

img_cnn = img_cnn()
img_cnn.cuda()
# Waveform CNN load
img_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/disease_classify_best_weight/img_cnn_weight_6_800_128_best.pt"))
img_optimizer = optim.Adam(img_cnn.parameters(), lr=learning_rate)


def concat_test(spt_data,img_data):
    spt_cnn.eval()
    img_cnn.eval()
    test_loss = 0.0
    predictions = np.zeros((int(spt_data.shape[0]), classes))
    predictions2 = np.zeros((int(spt_data.shape[0]), classes))

    #target = np.zeros((int(test_feature_label.shape[0]), 6))
    for j in range(int(spt_data.shape[0])):
        spt_input = torch.Tensor(spt_data[j:(j+1), :, :, :]).cuda()
        img_input = torch.Tensor(img_data[j:(j+1), :, :, :]).cuda()
        test_output = spt_cnn(spt_input)
        test_output2 = img_cnn(img_input)

        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        predictions2[j:j+1,:] = test_output2.detach().cpu().numpy()

    return predictions, predictions2


############################################# Ensemble Model

# spt_train = np.load('/home/iichsk/workspace/respiratory_classify/numpy_dataset/trainset/mfcc_trainset_1634028950.2965963.npy')
# spt_test = np.load('/home/iichsk/workspace/respiratory_classify/numpy_dataset/testset/mfcc_testset_1634636557.4533825.npy')

# img_train = np.load('/home/iichsk/workspace/respiratory_classify/numpy_dataset/trainset/waveform_trainset_1634028950.2965963.npy')
# img_test = np.load('/home/iichsk/workspace/respiratory_classify/numpy_dataset/testset/waveform_testset_1634636557.4533825.npy')

# label_train = np.load('/home/iichsk/workspace/respiratory_classify/numpy_dataset/trainset/train_label_1634028950.2965963.npy')
# label_test = np.load('/home/iichsk/workspace/respiratory_classify/numpy_dataset/testset/test_label_1634028950.2965963.npy')


# concat_train(spt_train,img_train,label_train)
predictions,predictions2 = concat_test(spt_test,img_test)
classpreds = np.argmax(predictions, axis=1)  # predicted classes
classpreds2 = np.argmax(predictions2, axis=1)  # predicted classes

print('classpreds', classpreds)
print('classpreds2', classpreds2)

target = np.argmax(label_test, axis=1)  # true classes
print('target', target)
c_names = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Pneumonia', 'URTI', 'Healthy']
# Classification Report
print(classification_report(target, classpreds, target_names=c_names))
# Confusion Matrix
print(confusion_matrix(target, classpreds))

print(classification_report(target, classpreds2, target_names=c_names))
print(confusion_matrix(target, classpreds2))
