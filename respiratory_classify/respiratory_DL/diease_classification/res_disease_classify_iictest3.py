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


batch_size = 128
learning_rate = 1e-3
filter_size1 = 10
filter_size2 = 2
classes = 6
spt_dropout = 0.325
img_dropout = 0.325
mypath = "/home/iichsk/workspace/dataset/iic_respiratory/filtered_wav_dwt/"
mypath2 = "/home/iichsk/workspace/dataset/iic_respiratory/filtered_wav_dwt/"

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]
filenames2 = [f for f in listdir(mypath2) if (isfile(join(mypath2, f)) and f.endswith('.png'))]


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

spt_features = []
img_features = []

def extract_image(file_name):
    im = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    return im

for file_name in filepaths:
    print('MFCC_file_name =',file_name)
    wave_file_name = file_name+'_waveform.png'
    print('wave_file_name =',wave_file_name)

    data = extract_features(file_name)
    spt_features.append(data)
    img_data = extract_image(wave_file_name)
    img_features.append(img_data)


print('Finished feature extraction from ', len(spt_features), ' files')
print('Finished feature extraction from ', len(img_features), ' files')

spt_features1 = np.array(spt_features)
img_features1 = np.array(img_features)
spt_features1 = np.reshape(spt_features1, (*spt_features1.shape, 1))
img_features1 = np.reshape(img_features1, (*img_features1.shape, 1))
spt_test = np.transpose(spt_features1, (0,3,1,2))
img_test = np.transpose(img_features1, (0,3,1,2))

# construct model on cuda if available
use_cuda = torch.cuda.is_available()

# loss
criterion = nn.MultiLabelSoftMarginLoss()
class spt_cnn(nn.Module):

    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
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

        return spt_x, spt_feature_x

spt_cnn = spt_cnn()
spt_cnn.cuda()
spt_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/spt_cnn_weight_6_600_128_best.pt"))
spt_optimizer = optim.Adam(spt_cnn.parameters(), lr=learning_rate)

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

        return img_x, img_feature_x
img_cnn = img_cnn()
img_cnn.cuda()
img_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/img_cnn_weight_6_800_128_best.pt"))

img_optimizer = optim.Adam(img_cnn.parameters(), lr=learning_rate)


# hyper-parameters
class concat_fc(nn.Module):
    def __init__(self):
        super(concat_fc, self).__init__()
       
        self.concat_fc1 = nn.Linear(384, 128)
        self.concat_fc2 = nn.Linear(128, classes)
        self.concat_relu = nn.ReLU()
        self.concat_bn1 = torch.nn.BatchNorm1d(128)
        self.concat_dropout1 = nn.Dropout(0.4)
    
    def forward(self, concat_x):
        
        concat_x = concat_x.view(concat_x.size(0), -1)
        concat_x = self.concat_fc1(concat_x)  #
        concat_x = self.concat_bn1(concat_x)
        concat_x = self.concat_dropout1(concat_x)
        concat_x = self.concat_fc2(concat_x)  #
        
        return concat_x


concat_fc = concat_fc()
concat_fc.cuda()
# backpropagation method
concat_fc.load_state_dict(torch.load("/home/iichsk/workspace/weight/concat_fc_weight_450_128_best.pt"))
concat_optimizer = optim.Adam(concat_fc.parameters(), lr=learning_rate)

def spt_cnn_test(spt_data):
    spt_cnn.eval()
    test_loss = 0.0
    predictions = np.zeros((int(spt_data.shape[0]), classes))
    spt_feature = np.zeros((int(spt_data.shape[0]), 128, 1,1))

    for j in range(int(spt_data.shape[0])):
        spt_input = torch.Tensor(spt_data[j:(j+1), :, :, :]).cuda()
        test_output, spt_feature_x = spt_cnn(spt_input)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        spt_feature[j:j+1,:,:,:] = spt_feature_x.data.cpu().numpy()

    return predictions, spt_feature

def img_cnn_test(img_data):
    img_cnn.eval()
    test_loss = 0.0
    predictions = np.zeros((int(img_data.shape[0]), classes))
    img_feature = np.zeros((int(img_data.shape[0]), 256, 1,1))

    for j in range(int(img_data.shape[0])):
        img_input = torch.Tensor(img_data[j:(j+1), :, :, :]).cuda()
        test_output, img_feature_x = img_cnn(img_input)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        img_feature[j:j+1,:,:,:] = img_feature_x.data.cpu().numpy()

    return predictions, img_feature

def concat_test(concat_testset):
    concat_fc.eval()
    predictions = np.zeros((int(concat_testset.shape[0]), classes))

    for j in range(int(concat_testset.shape[0])):
        concat_input = torch.Tensor(concat_testset[j:(j+1), :, :, :]).cuda()
        test_output = concat_fc(concat_input)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        
    return predictions


#spectrum part
spt_prediction, spt_feature = spt_cnn_test(spt_test)
classpreds = np.argmax(spt_prediction, axis=1)  # predicted classes
print('MFCC classprediction= ')
print(classpreds)

#waveform part
img_prediction, img_feature = img_cnn_test(img_test)
classpreds = np.argmax(img_prediction, axis=1)  # predicted classes
print('waveform classprediction= ')
print(classpreds)

#concat data part
concat_testset = np.concatenate((spt_feature, img_feature),1)
predictions = concat_test(concat_testset)
classpreds = np.argmax(predictions, axis=1)  # predicted classes
print('concat classprediction= ', classpreds)

