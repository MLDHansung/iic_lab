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
classes = 4
MFCC_dropout = 0.3
scalo_dropout = 0.3
wave_dropout = 0.3

ensem_num_epochs = 300


batch_size = 256
learning_rate = 1e-3
classes = 2
MFCC_dropout = 0.3
mfcc_num_epochs = 1000

mfcc_trainset=np.load('mfcc_trainset_2class.npy')
mfcc_testset=np.load('mfcc_testset_2class.npy')
train_label=np.load('train_label_2class.npy')
test_label=np.load('test_label_2class.npy')

use_cuda = torch.cuda.is_available()

# loss
criterion = nn.MultiLabelSoftMarginLoss()
# MFCC CNN model
class MFCC_cnn(nn.Module):

    def __init__(self):

        super(MFCC_cnn, self).__init__()

        self.MFCC_conv1 = nn.Conv2d(1, 16, kernel_size=5)  # 
        self.MFCC_bn1 = torch.nn.BatchNorm2d(16)
        self.MFCC_pool1 = nn.MaxPool2d(2)  #
        self.MFCC_conv2 = nn.Conv2d(16, 32, kernel_size=3)  # 
        self.MFCC_bn2 = torch.nn.BatchNorm2d(32)
        self.MFCC_pool2 = nn.MaxPool2d(2)  # 
        self.MFCC_dropout2 = nn.Dropout(MFCC_dropout)
        self.MFCC_conv3 = nn.Conv2d(32, 64, kernel_size=(1,2))  # 
        self.MFCC_bn3 = torch.nn.BatchNorm2d(64)
        self.MFCC_pool3 = nn.MaxPool2d(2)  # 
        self.MFCC_dropout3 = nn.Dropout(MFCC_dropout)
        self.MFCC_conv4 = nn.Conv2d(64, 128, kernel_size=(1,2))  # 
        self.MFCC_bn4 = torch.nn.BatchNorm2d(128)
        self.MFCC_pool4 = nn.MaxPool2d(2)  # 
        self.MFCC_conv5 = nn.Conv2d(128, 256, kernel_size=(1,2))  # 
        self.MFCC_bn5 = torch.nn.BatchNorm2d(256)
        self.MFCC_pool5 = nn.MaxPool2d(2)  # 


        self.MFCC_global_pool = nn.AdaptiveAvgPool2d(1)
        # self.MFCC_bn0 = torch.nn.BatchNorm2d(128)
        self.MFCC_fc1 = nn.Linear(256, 128)
        self.MFCC_fc_bn1 = torch.nn.BatchNorm1d(128)
        self.MFCC_fc2 = nn.Linear(128, classes)  
        self.MFCC_relu = nn.ReLU()
    
    def forward(self, MFCC_x):

        MFCC_x = self.MFCC_relu(self.MFCC_bn1(self.MFCC_conv1(MFCC_x)))
        MFCC_x = self.MFCC_pool1(MFCC_x)  #
        MFCC_x = self.MFCC_relu(self.MFCC_bn2(self.MFCC_conv2(MFCC_x)))
        MFCC_x = self.MFCC_pool2(MFCC_x)  #
        MFCC_x = self.MFCC_relu(self.MFCC_bn3(self.MFCC_conv3(MFCC_x)))
        MFCC_x = self.MFCC_pool3(MFCC_x)  #
        MFCC_x = self.MFCC_relu(self.MFCC_bn4(self.MFCC_conv4(MFCC_x)))
        MFCC_x = self.MFCC_pool4(MFCC_x)  #
        MFCC_x = self.MFCC_relu(self.MFCC_bn5(self.MFCC_conv5(MFCC_x)))
        MFCC_x = self.MFCC_pool5(MFCC_x)  #

        MFCC_x = self.MFCC_global_pool(MFCC_x)
        MFCC_x = MFCC_x.view(MFCC_x.size(0), -1) # 
        MFCC_x = self.MFCC_fc1(MFCC_x) 
        MFCC_feature_raw = MFCC_x
        MFCC_x = self.MFCC_fc_bn1(MFCC_x)
        MFCC_feature_x = MFCC_x

        MFCC_x = self.MFCC_fc2(MFCC_x)

        return MFCC_x, MFCC_feature_x, MFCC_feature_raw


MFCC_cnn = MFCC_cnn()
MFCC_cnn.cuda()
MFCC_optimizer = optim.Adam(MFCC_cnn.parameters(), lr=learning_rate)
MFCC_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/MFCC_cnn2_weight_200_256_abnormal.pt"))


# MFCC CNN test
def MFCC_cnn_test(mfcc_testset, label):
    MFCC_cnn.eval()
    test_loss = 0.0
    predictions = np.zeros((int(mfcc_testset.shape[0]), classes))
    mfcc_features = np.zeros((int(mfcc_testset.shape[0]), 128))
    mfcc_features_raw = np.zeros((int(mfcc_testset.shape[0]), 128))
    # test_embeddings = torch.zeros((0, 1310), dtype=torch.float32)
    for j in range(int(mfcc_testset.shape[0])):
        mfcc_input = torch.Tensor(mfcc_testset[j:(j+1), :, :, :]).cuda()
        test_output, mfcc_feature, mfcc_feature_raw = MFCC_cnn(mfcc_input)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        mfcc_features[j:j+1,:] = mfcc_feature.data.cpu().numpy()
        mfcc_features_raw[j:j+1,:] = mfcc_feature_raw.data.cpu().numpy()
        # test_embeddings = torch.cat((test_embeddings,mfcc_feature.detach().cpu()),0)
    return predictions, mfcc_features, mfcc_features_raw


# MFCC CNN part
MFCC_predictions, MFCC_test_features, MFCC_features_raw = MFCC_cnn_test(mfcc_testset, test_label)
count_t = time.time()

# CNN models report
MFCC_predictions_arg = np.argmax(MFCC_predictions, axis=1)  
target = np.argmax(test_label, axis=1)  # true classes
c_names = ['normal','symptom']

print('#'*10,'MFCC CNN report','#'*10)
print(classification_report(target, MFCC_predictions_arg, target_names=c_names))
print(confusion_matrix(target, MFCC_predictions_arg))


# dataset load
mypath = "/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/processed_audio_files_nopad/" # respiratory sound directory
p_diag = pd.read_csv("/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/csv_data/3classes_label.csv",header=None) # patient diagnosis csv file

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

# preprocess trainset
mfcc_trainset=[]
waveform_trainset=[]
scalogram_trainset=[]
for p_id in trainset_p_id:
    for file_name in listdir(mypath):
        if (int(file_name.split('_')[0]) == p_id):

            extract_mfcc_data = extract_MFCC((mypath+file_name))
            mfcc_trainset.append(extract_mfcc_data)

mfcc_trainset = np.array(mfcc_trainset)

# preprocess testset
mfcc_testset=[]
for p_id in testset_p_id:
    for file_name in listdir(mypath):
        if (int(file_name.split('_')[0]) == p_id):

            extract_mfcc_data = extract_MFCC((mypath+file_name))
            mfcc_testset.append(extract_mfcc_data)
            p_dig[0]==int(file_name.split('_')[0])
            labels.append()
 
labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])

print('Finished feature extraction from ', len(dataset_p_id), ' files')
unique_elements, counts_elements = np.unique(labels, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

le = LabelEncoder()
i_labels = le.fit_transform(labels)
oh_labels = to_categorical(i_labels)

mfcc_testset = np.array(mfcc_testset)

mfcc_trainset = np.reshape(mfcc_trainset, (*mfcc_trainset.shape, 1))
mfcc_testset = np.reshape(mfcc_testset, (*mfcc_testset.shape, 1))

mfcc_trainset = np.transpose(mfcc_trainset, (0,3,1,2))
mfcc_testset = np.transpose(mfcc_testset, (0,3,1,2))

np.save('mfcc_trainset_3classes.npy', mfcc_trainset)
np.save('mfcc_testset_3classes.npy', mfcc_testset)
np.save('train_label_3classes.npy', train_label)
np.save('test_label_3classes.npy', test_label)

# # construct model on cuda if available
use_cuda = torch.cuda.is_available()

# loss
criterion = nn.MultiLabelSoftMarginLoss()
# MFCC CNN model
class MFCC_cnn(nn.Module):

    def __init__(self):

        super(MFCC_cnn, self).__init__()

        self.MFCC_conv1 = nn.Conv2d(1, 16, kernel_size=5)  # 
        self.MFCC_bn1 = torch.nn.BatchNorm2d(16)
        self.MFCC_pool1 = nn.MaxPool2d(2)  #
        self.MFCC_conv2 = nn.Conv2d(16, 32, kernel_size=3)  # 
        self.MFCC_bn2 = torch.nn.BatchNorm2d(32)
        self.MFCC_pool2 = nn.MaxPool2d(2)  # 
        self.MFCC_dropout2 = nn.Dropout(MFCC_dropout)
        self.MFCC_conv3 = nn.Conv2d(32, 64, kernel_size=(1,2))  # 
        self.MFCC_bn3 = torch.nn.BatchNorm2d(64)
        self.MFCC_pool3 = nn.MaxPool2d(2)  # 
        self.MFCC_dropout3 = nn.Dropout(MFCC_dropout)
        self.MFCC_conv4 = nn.Conv2d(64, 128, kernel_size=(1,2))  # 
        self.MFCC_bn4 = torch.nn.BatchNorm2d(128)
        self.MFCC_pool4 = nn.MaxPool2d(2)  # 
        self.MFCC_conv5 = nn.Conv2d(128, 256, kernel_size=(1,2))  # 
        self.MFCC_bn5 = torch.nn.BatchNorm2d(256)
        self.MFCC_pool5 = nn.MaxPool2d(2)  # 


        self.MFCC_global_pool = nn.AdaptiveAvgPool2d(1)
        self.MFCC_fc1 = nn.Linear(6400, 128)
        self.MFCC_fc_bn1 = torch.nn.BatchNorm1d(128)
        self.MFCC_relu = nn.ReLU()
    
    def forward(self, MFCC_x):

        MFCC_x = self.MFCC_relu(self.MFCC_bn1(self.MFCC_conv1(MFCC_x)))
        MFCC_x = self.MFCC_pool1(MFCC_x)  #
        MFCC_x = self.MFCC_relu(self.MFCC_bn2(self.MFCC_conv2(MFCC_x)))
        MFCC_x = self.MFCC_pool2(MFCC_x)  #
        MFCC_x = self.MFCC_relu(self.MFCC_bn3(self.MFCC_conv3(MFCC_x)))
        MFCC_x = self.MFCC_pool3(MFCC_x)  #
        MFCC_x = self.MFCC_relu(self.MFCC_bn4(self.MFCC_conv4(MFCC_x)))
        MFCC_x = self.MFCC_pool4(MFCC_x)  #
        MFCC_x = self.MFCC_relu(self.MFCC_bn5(self.MFCC_conv5(MFCC_x)))
        MFCC_x = self.MFCC_pool5(MFCC_x)  #

        MFCC_x = MFCC_x.view(MFCC_x.size(0), -1) #
        MFCC_x = self.MFCC_fc1(MFCC_x) 
        MFCC_feature_raw = MFCC_x
        MFCC_x = self.MFCC_fc_bn1(MFCC_x) # feature size 128 save to excell
        MFCC_feature_x = MFCC_x


        return MFCC_feature_x




class concat_fc(nn.Module):
    def __init__(self):
        super(concat_fc, self).__init__()

        self.concat_fc1 = nn.Linear(384, 128)
        self.concat_fc2 = nn.Linear(128, 64)
        self.concat_fc3 = nn.Linear(64, classes)
        self.concat_bn1 = torch.nn.BatchNorm1d(128)
        self.concat_bn2 = torch.nn.BatchNorm1d(64)

    def forward(self, MFCC_input,scalo_input,wave_input):   
    
        MFCC_feature = MFCC_cnn(MFCC_input)
        scalo_feature = scalo_cnn(scalo_input)
        wave_feature = wave_cnn(wave_input)
        concat_x = torch.cat((MFCC_feature, scalo_feature,wave_feature),1)
        concat_x = self.concat_fc1(concat_x) 
        # concat_x = self.concat_bn1(concat_x)
        concat_x = self.concat_fc2(concat_x) 
        # concat_x = self.concat_bn2(concat_x)
        concat_x = self.concat_fc3(concat_x) 

        return concat_x


MFCC_cnn = MFCC_cnn()
MFCC_cnn.cuda()
MFCC_optimizer = optim.Adam(MFCC_cnn.parameters(), lr=learning_rate)
MFCC_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/MFCC_cnn2_weight_1000_128_1631600870.018815.pt"),strict=False)

scalo_cnn = scalo_cnn()
scalo_cnn.cuda()
scalo_optimizer = optim.Adam(scalo_cnn.parameters(), lr=learning_rate)
scalo_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/Scalogram_cnn_weight_1000_128_1631521741.4954169.pt"),strict=False)

wave_cnn = wave_cnn()
wave_cnn.cuda()
wave_optimizer = optim.Adam(wave_cnn.parameters(), lr=learning_rate)
wave_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/Waveform_cnn_weight_1000_128_1631515583.3403041.pt"),strict=False)

concat_fc = concat_fc()
concat_fc.cuda()
concat_optimizer = optim.Adam(concat_fc.parameters(), lr=learning_rate)


# Ensemble train
def concat_train(mfcc_trainset,scalogram_trainset,waveform_trainset,feature_label):
    MFCC_cnn.eval()
    scalo_cnn.eval()
    wave_cnn.eval()

    z = np.random.permutation(mfcc_trainset.shape[0])
    trn_loss_list = []
    print('concat train start!!!!!')
    concat_fc.train()
    for epoch in range(ensem_num_epochs):
        trn_loss = 0.0
        for i in range(int(mfcc_trainset.shape[0] / batch_size)):
            mfcc_input = torch.Tensor(mfcc_trainset[z[i*batch_size:(i+1)* batch_size], :, :, :]).cuda()
            scalo_input = torch.Tensor(scalogram_trainset[z[i*batch_size:(i+1)* batch_size], :, :, :]).cuda()
            wave_input = torch.Tensor(waveform_trainset[z[i*batch_size:(i+1)* batch_size], :, :, :]).cuda()

            label_torch =  torch.Tensor(feature_label[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            # grad init
            concat_optimizer.zero_grad()
            model_output = concat_fc(mfcc_input,scalo_input,wave_input)
            # calculate loss
            concat_loss = criterion(model_output, label_torch)
            # back propagation
            concat_loss.backward(retain_graph=True)
            # weight update
            concat_optimizer.step()
            # trn_loss summary
            trn_loss += concat_loss.item()      
            # 학습과정 출력
        if (epoch + 1) % 50 == 0:  #
            print("epoch: {}/{} | trn loss: {:.8f}".format(
                epoch + 1, ensem_num_epochs, trn_loss / 100))
            trn_loss_list.append(trn_loss / 100)
    count_t = time.time()
    torch.save(concat_fc.state_dict(), "/home/iichsk/workspace/weight/concat_fc_weight_{}_{}_{}.pt".format(ensem_num_epochs, batch_size, count_t))

# Ensemble test
def concat_test(mfcc_testset,scalogram_testset,waveform_testset):
    MFCC_cnn.eval()
    scalo_cnn.eval()
    wave_cnn.eval()

    concat_fc.eval()
    test_loss = 0.0
    predictions = np.zeros((int(mfcc_testset.shape[0]), classes))

    for j in range(int(mfcc_testset.shape[0])):
        mfcc_input = torch.Tensor(mfcc_testset[j:(j+1), :, :, :]).cuda()
        scalo_input = torch.Tensor(scalogram_testset[j:(j+1), :, :, :]).cuda()
        wave_input = torch.Tensor(waveform_testset[j:(j+1), :, :, :]).cuda()

        test_output = concat_fc(mfcc_input,scalo_input,wave_input)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()

    return predictions




# Ensemble part
# trainset feature concatenate and Ensemble train
concat_train(mfcc_trainset,scalogram_trainset,waveform_trainset,train_label)

# testset feature concatenate and Ensemble test
concat_testset = np.concatenate((MFCC_test_features, scalo_test_features, wave_test_features),1)
ensem_predictions = concat_test(concat_testset)

# Ensemble model report
ensem_predictions_arg = np.argmax(ensem_predictions, axis=1)  
print('#'*10,'Triple Ensemble CNN1 report','#'*10)
print(classification_report(target, ensem_predictions_arg, target_names=c_names))
print(confusion_matrix(target, ensem_predictions_arg))



