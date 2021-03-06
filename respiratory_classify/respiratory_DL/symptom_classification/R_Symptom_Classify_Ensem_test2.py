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
mfcc_num_epochs = 2000
wave_num_epochs = 1000
scalo_num_epochs = 1000
ensem_num_epochs = 300

# dataset load
mypath = "/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/processed_audio_files_nopad/" # respiratory sound directory
mypath_waveform = "/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/waveform_image/" # waveform image directory
p_diag = pd.read_csv("/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/csv_data/data2.csv",header=None) # patient diagnosis csv file

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]

p_id_in_file = [] # patient IDs corresponding to each file
for name in filenames:
    p_id_in_file.append(int(name.split('_')[0]))
p_id_in_file = np.array(p_id_in_file)
dataset_p_id = p_id_in_file

labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])

print('Finished feature extraction from ', len(dataset_p_id), ' files')
unique_elements, counts_elements = np.unique(labels, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

le = LabelEncoder()
i_labels = le.fit_transform(labels)
oh_labels = to_categorical(i_labels)

trainset_p_id, testset_p_id, train_label, test_label = train_test_split(dataset_p_id, oh_labels, stratify=i_labels,
                                                    test_size=0.2, random_state = 42)


# def extract_MFCC(file_name):
#     """
#     This function takes in the path for an audio file as a string, loads it, and returns the MFCC
#     of the audio"""
#     max_pad_len = 862
#     try:
#         audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=20)
#         mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
#         pad_width = max_pad_len - mfccs.shape[1]
#         mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
#         #mfccs = np.pad(mfccs, pad_width=((0,0)), mode='constant')
#     except Exception as e:
#         print("Error encountered while parsing file: ", file_name)
#         return None

#     return mfccs

# def extract_scalogram(file_name):
#     """
#     This function takes in the path for an audio file as a string, loads it, and returns the scalogram
#     of the audio"""
#     max_pad_len = 1393
#     try:
#         clip, sample_rate = librosa.load(file_name, mono=True) 
#         clip_cqt = librosa.cqt(clip, hop_length=256, sr=sample_rate, fmin=30, bins_per_octave=32, n_bins=150, filter_scale=1.)    
#         clip_cqt_abs = np.abs(clip_cqt)
#         pad_width = max_pad_len - clip_cqt_abs.shape[1]
#         clip_cqt_abs = np.pad(clip_cqt_abs, pad_width=((0, 0), (0, pad_width)), mode='constant')
#         times = np.arange(len(clip))/float(sample_rate)
#     except Exception as e:
#         print("Error encountered while parsing file: ", file_name)
#         return None 
     
#     return clip_cqt_abs


# # preprocess trainset
# mfcc_trainset=[]
# waveform_trainset=[]
# scalogram_trainset=[]
# for p_id in trainset_p_id:
#     for file_name in listdir(mypath):
#         if (int(file_name.split('_')[0]) == p_id):

#             extract_mfcc_data = extract_MFCC((mypath+file_name))
#             mfcc_trainset.append(extract_mfcc_data)

#             extract_scalogram_data = extract_scalogram((mypath+file_name))
#             scalogram_trainset.append(extract_scalogram_data)

#     for file_name_wave in listdir(mypath_waveform):
#         if (int(file_name_wave.split('_')[0]) == p_id):
#             extract_waveform_data = cv2.imread(mypath_waveform+file_name_wave, cv2.IMREAD_GRAYSCALE)
#             waveform_trainset.append(extract_waveform_data)

# mfcc_trainset = np.array(mfcc_trainset)


# preprocess testset
mfcc_testset=[]
p_id_testset=[]
for p_id in testset_p_id:
    for file_name in listdir(mypath):
        if (int(file_name.split('_')[0]) == p_id):

            p_id_testset.append(file_name)
p_id_testset=pd.DataFrame(p_id_testset)
p_id_testset.to_csv('p_id_testset.csv')


mfcc_trainset = np.load('/home/iichsk/workspace/trainset/mfcc_trainset.npy')
mfcc_testset = np.load('/home/iichsk/workspace/testset/mfcc_testset.npy')
waveform_trainset = np.load('/home/iichsk/workspace/trainset/waveform_trainset.npy')
waveform_testset = np.load('/home/iichsk/workspace/testset/waveform_testset.npy')
scalogram_trainset = np.load('/home/iichsk/workspace/trainset/scalogram_trainset.npy')
scalogram_testset = np.load('/home/iichsk/workspace/testset/scalogram_testset.npy')
train_label = np.load('/home/iichsk/workspace/trainset/train_label.npy')
test_label = np.load('/home/iichsk/workspace/testset/test_label.npy')

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

        MFCC_x = MFCC_x.view(MFCC_x.size(0), -1) #
        MFCC_x = self.MFCC_fc1(MFCC_x) 
        MFCC_feature_raw = MFCC_x
        MFCC_x = self.MFCC_fc_bn1(MFCC_x) # feature size 128 save to excell
        MFCC_feature_x = MFCC_x

        MFCC_x = self.MFCC_fc2(MFCC_x)

        return MFCC_x, MFCC_feature_x, MFCC_feature_raw

class scalo_cnn(nn.Module):
    def __init__(self):

        super(scalo_cnn, self).__init__()

        self.scalo_conv1 = nn.Conv2d(1, 16, kernel_size=4)  
        self.scalo_pool1 = nn.MaxPool2d(2)  
        self.scalo_dropout1 = nn.Dropout(scalo_dropout)
        self.scalo_conv2 = nn.Conv2d(16, 32, kernel_size=2)  
        self.scalo_pool2 = nn.MaxPool2d(2)  
        self.scalo_dropout2 = nn.Dropout(scalo_dropout)
        self.scalo_conv3 = nn.Conv2d(32, 64, kernel_size=2)  
        self.scalo_pool3 = nn.MaxPool2d(2)  
        self.scalo_dropout3 = nn.Dropout(scalo_dropout)
        self.scalo_conv4 = nn.Conv2d(64, 128, kernel_size=2)  
        self.scalo_pool4 = nn.MaxPool2d(2)  
        self.scalo_dropout4 = nn.Dropout(scalo_dropout)
        self.scalo_conv5 = nn.Conv2d(128, 256, kernel_size=2)  
        self.scalo_pool5 = nn.MaxPool2d(2)  
        self.scalo_dropout5 = nn.Dropout(scalo_dropout)        
        self.scalo_global_pool = nn.AdaptiveAvgPool2d(1)
        self.scalo_fc1 = nn.Linear(256, 128)
        self.scalo_bn1 = torch.nn.BatchNorm1d(128)
        self.scalo_fcdropout1 = nn.Dropout(scalo_dropout)
        self.scalo_fc2 = nn.Linear(128, classes)       
        self.scalo_relu = nn.ReLU()
    
    def forward(self, scalo_x):

        scalo_x = self.scalo_relu(self.scalo_conv1(scalo_x))
        scalo_x = self.scalo_pool1(scalo_x) 
        scalo_x = self.scalo_dropout1(scalo_x)
        scalo_x = self.scalo_relu(self.scalo_conv2(scalo_x))
        scalo_x = self.scalo_pool2(scalo_x) 
        scalo_x = self.scalo_dropout2(scalo_x)
        scalo_x = self.scalo_relu(self.scalo_conv3(scalo_x))
        scalo_x = self.scalo_pool3(scalo_x) 
        scalo_x = self.scalo_dropout3(scalo_x) 
        scalo_x = self.scalo_relu(self.scalo_conv4(scalo_x))
        scalo_x = self.scalo_pool4(scalo_x) 
        scalo_x = self.scalo_dropout4(scalo_x) 
        scalo_x = self.scalo_relu(self.scalo_conv5(scalo_x))
        scalo_x = self.scalo_pool5(scalo_x) 
        scalo_x = self.scalo_dropout5(scalo_x)    
        scalo_x = self.scalo_global_pool(scalo_x) 
        scalo_x = scalo_x.view(scalo_x.size(0), -1) 
        scalo_x = self.scalo_fc1(scalo_x) 
        scalo_x = self.scalo_relu(scalo_x)
        scalo_x = self.scalo_bn1(scalo_x)
        scalo_feature_x = scalo_x # feature size 128 save to excell
        scalo_x = self.scalo_fcdropout1(scalo_x)
        scalo_x = self.scalo_fc2(scalo_x)

        return scalo_x, scalo_feature_x

class wave_cnn(nn.Module):
    def __init__(self):

        super(wave_cnn, self).__init__()

        self.wave_conv1 = nn.Conv2d(1, 16, kernel_size=(2,10))  
        self.wave_pool1 = nn.MaxPool2d(2)  
        self.wave_dropout1 = nn.Dropout(wave_dropout)
        self.wave_conv2 = nn.Conv2d(16, 32, kernel_size=(2,10))  
        self.wave_pool2 = nn.MaxPool2d(2) 
        self.wave_dropout2 = nn.Dropout(wave_dropout)
        self.wave_conv3 = nn.Conv2d(32, 64, kernel_size=(2,10))  
        self.wave_pool3 = nn.MaxPool2d(2)  
        self.wave_dropout3 = nn.Dropout(wave_dropout)
        self.wave_conv4 = nn.Conv2d(64, 128, kernel_size=2)  
        self.wave_pool4 = nn.MaxPool2d(2)  
        self.wave_dropout4 = nn.Dropout(wave_dropout)
        self.wave_conv5 = nn.Conv2d(128, 256, kernel_size=2)  
        self.wave_pool5 = nn.MaxPool2d(2)  
        self.wave_dropout5 = nn.Dropout(wave_dropout)
        self.wave_conv6 = nn.Conv2d(256, 512, kernel_size=2) 
        self.wave_pool6 = nn.MaxPool2d(2)  
        self.wave_dropout6 = nn.Dropout(wave_dropout)
        self.wave_global_pool = nn.AdaptiveAvgPool2d(1)
        self.wave_fc1 = nn.Linear(512, 256)
        self.wave_fc2 = nn.Linear(256, 128)
        self.wave_bn2 = torch.nn.BatchNorm1d(128)
        self.wave_fc3 = nn.Linear(128, classes)
        self.wave_relu = nn.ReLU()
       
    
    def forward(self, wave_x):
        
        wave_x = self.wave_relu(self.wave_conv1(wave_x))
        wave_x = self.wave_pool1(wave_x) 
        wave_x = self.wave_dropout1(wave_x)
        wave_x = self.wave_relu(self.wave_conv2(wave_x))
        wave_x = self.wave_pool2(wave_x) 
        wave_x = self.wave_dropout2(wave_x)
        wave_x = self.wave_relu(self.wave_conv3(wave_x))
        wave_x = self.wave_pool3(wave_x) 
        wave_x = self.wave_dropout3(wave_x) 
        wave_x = self.wave_relu(self.wave_conv4(wave_x))
        wave_x = self.wave_pool4(wave_x) 
        wave_x = self.wave_dropout4(wave_x) 
        wave_x = self.wave_relu(self.wave_conv5(wave_x))
        wave_x = self.wave_pool5(wave_x) 
        wave_x = self.wave_dropout5(wave_x) 
        wave_x = self.wave_relu(self.wave_conv6(wave_x))
        wave_x = self.wave_pool6(wave_x) 
        wave_x = self.wave_dropout6(wave_x) 
        
        wave_x = self.wave_global_pool(wave_x)
        wave_x = wave_x.view(wave_x.size(0), -1)
        wave_x = self.wave_fc1(wave_x) 
        wave_x = self.wave_fc2(wave_x)
        wave_feature_x = self.wave_bn2(wave_x) # feature size 128 save to excell
        wave_x = self.wave_fc3(wave_x)

        return wave_x, wave_feature_x

class concat_fc(nn.Module):
    def __init__(self):
        super(concat_fc, self).__init__()

        self.concat_fc1 = nn.Linear(384, 128)
        self.concat_fc2 = nn.Linear(128, 64)
        self.concat_fc3 = nn.Linear(64, classes)
        self.concat_bn1 = torch.nn.BatchNorm1d(128)
        self.concat_bn2 = torch.nn.BatchNorm1d(64)

    def forward(self, concat_input):   
    
        
        concat_x = self.concat_fc1(concat_input) 
        # concat_x = self.concat_bn1(concat_x)
        concat_x = self.concat_fc2(concat_x) 
        # concat_x = self.concat_bn2(concat_x)
        concat_x = self.concat_fc3(concat_x) 

        return concat_x


MFCC_cnn = MFCC_cnn()
MFCC_cnn.cuda()
MFCC_optimizer = optim.Adam(MFCC_cnn.parameters(), lr=learning_rate)
MFCC_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/MFCC_cnn2_weight_1000_128_1631600870.018815.pt"))

scalo_cnn = scalo_cnn()
scalo_cnn.cuda()
scalo_optimizer = optim.Adam(scalo_cnn.parameters(), lr=learning_rate)
scalo_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/Scalogram_cnn_weight_1000_128_1631521741.4954169.pt"))

wave_cnn = wave_cnn()
wave_cnn.cuda()
wave_optimizer = optim.Adam(wave_cnn.parameters(), lr=learning_rate)
wave_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/Waveform_cnn_weight_1000_128_1631515583.3403041.pt"))

concat_fc = concat_fc()
concat_fc.cuda()
concat_optimizer = optim.Adam(concat_fc.parameters(), lr=learning_rate)
concat_fc.load_state_dict(torch.load("/home/iichsk/workspace/weight/concat_fc_weight_350_128_1627272809.0676024.pt"))


# Ensemble train
def concat_train(concat_trainset,feature_label):
    
    z = np.random.permutation(concat_trainset.shape[0])
    trn_loss_list = []
    print('concat train start!!!!!')
    concat_fc.train()
    for epoch in range(ensem_num_epochs):
        trn_loss = 0.0
        for i in range(int(concat_trainset.shape[0] / batch_size)):
            concat_input = torch.Tensor(concat_trainset[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            label_torch =  torch.Tensor(feature_label[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            # grad init
            concat_optimizer.zero_grad()
            model_output = concat_fc(concat_input)
            # calculate loss
            concat_loss = criterion(model_output, label_torch)
            # back propagation
            concat_loss.backward(retain_graph=True)
            # weight update
            concat_optimizer.step()
            # trn_loss summary
            trn_loss += concat_loss.item()      
            # ???????????? ??????
        if (epoch + 1) % 50 == 0:  #
            print("epoch: {}/{} | trn loss: {:.8f}".format(
                epoch + 1, ensem_num_epochs, trn_loss / 100))
            trn_loss_list.append(trn_loss / 100)
    count_t = time.time()
    torch.save(concat_fc.state_dict(), "/home/iichsk/workspace/weight/concat_fc_weight_{}_{}_{}.pt".format(ensem_num_epochs, batch_size, count_t))

# MFCC CNN test
def MFCC_cnn_test(mfcc_testset, label):
    MFCC_cnn.eval()
    test_loss = 0.0
    predictions = np.zeros((int(mfcc_testset.shape[0]), classes))
    mfcc_features = np.zeros((int(mfcc_testset.shape[0]), 128))
    mfcc_features_raw = np.zeros((int(mfcc_testset.shape[0]), 128))

    for j in range(int(mfcc_testset.shape[0])):
        mfcc_input = torch.Tensor(mfcc_testset[j:(j+1), :, :, :]).cuda()
        test_output, mfcc_feature, mfcc_feature_raw = MFCC_cnn(mfcc_input)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        mfcc_features[j:j+1,:] = mfcc_feature.data.cpu().numpy()
        mfcc_features_raw[j:j+1,:] = mfcc_feature_raw.data.cpu().numpy()

    return predictions, mfcc_features, mfcc_features_raw

# Scalogram CNN test
def scalo_cnn_test(scalogram_testset, label):
    scalo_cnn.eval()
    test_loss = 0.0
    predictions = np.zeros((int(scalogram_testset.shape[0]), classes))
    scalo_features = np.zeros((int(scalogram_testset.shape[0]), 128))

    for j in range(int(scalogram_testset.shape[0])):
        scalo_input = torch.Tensor(scalogram_testset[j:(j+1), :, :, :]).cuda()
        test_output, scalo_feature = scalo_cnn(scalo_input)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        scalo_features[j:j+1,:] = scalo_feature.data.cpu().numpy()

    return predictions, scalo_features

# Waveform CNN test
def wave_cnn_test(waveform_testset, label):
    wave_cnn.eval()
    test_loss = 0.0
    predictions = np.zeros((int(waveform_testset.shape[0]), classes))
    wave_features = np.zeros((int(waveform_testset.shape[0]), 128))

    for j in range(int(waveform_testset.shape[0])):
        wave_input = torch.Tensor(waveform_testset[j:(j+1), :, :, :]).cuda()
        test_output, wave_feature = wave_cnn(wave_input)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        wave_features[j:j+1,:] = wave_feature.data.cpu().numpy()

    return predictions, wave_features

# Ensemble test
def concat_test(concat_testset):
    concat_fc.eval()
    test_loss = 0.0
    predictions = np.zeros((int(concat_testset.shape[0]), classes))

    for j in range(int(concat_testset.shape[0])):
        concat_input = torch.Tensor(concat_testset[j:(j+1), :]).cuda()
        test_output = concat_fc(concat_input)

        predictions[j:j+1,:] = test_output.detach().cpu().numpy()

    return predictions


# MFCC CNN part
_, MFCC_train_features, _ = MFCC_cnn_test(mfcc_trainset, train_label)
MFCC_predictions, MFCC_test_features, _ = MFCC_cnn_test(mfcc_testset, test_label)

# Scalogram CNN part
_, scalo_train_features = scalo_cnn_test(scalogram_trainset, train_label)
scalo_predictions, scalo_test_features = scalo_cnn_test(scalogram_testset, test_label)

# Waveform CNN part
_, wave_train_features = wave_cnn_test(waveform_trainset, train_label)
wave_predictions, wave_test_features = wave_cnn_test(waveform_testset, test_label)

# CNN models report
MFCC_predictions_arg = np.argmax(MFCC_predictions, axis=1)  
scalo_predictions_arg = np.argmax(scalo_predictions, axis=1) 
wave_predictions_arg = np.argmax(wave_predictions, axis=1)  
target = np.argmax(test_label, axis=1)  # true classes
c_names = ['normal','wheezes','crackles','Both']
print('#'*10,'MFCC CNN report','#'*10)
print(classification_report(target, MFCC_predictions_arg, target_names=c_names))
print(confusion_matrix(target, MFCC_predictions_arg))
print('#'*10,'Scalogram CNN report','#'*10)
print(classification_report(target, scalo_predictions_arg, target_names=c_names))
print(confusion_matrix(target, scalo_predictions_arg))
print('#'*10,'Waveform CNN report','#'*10)
print(classification_report(target, wave_predictions_arg, target_names=c_names))
print(confusion_matrix(target, wave_predictions_arg))

# Ensemble part
# trainset feature concatenate and Ensemble train
concat_trainset = np.concatenate((MFCC_train_features, scalo_train_features, wave_train_features),1)
print('concat_trainset.shape',concat_trainset.shape)
# concat_train(concat_trainset,train_label)

# testset feature concatenate and Ensemble test
concat_testset = np.concatenate((MFCC_test_features, scalo_test_features, wave_test_features),1)
ensem_predictions = concat_test(concat_testset)

# Ensemble model report
ensem_predictions_arg = np.argmax(ensem_predictions, axis=1)  
print('#'*10,'Triple Ensemble CNN1 report','#'*10)
print(classification_report(target, ensem_predictions_arg, target_names=c_names))
print(confusion_matrix(target, ensem_predictions_arg))

# result save to excell
MFCC_predictions_arg=np.expand_dims(MFCC_predictions_arg, axis=1)
scalo_predictions_arg=np.expand_dims(scalo_predictions_arg, axis=1)
wave_predictions_arg=np.expand_dims(wave_predictions_arg, axis=1)
ensem_predictions_arg=np.expand_dims(ensem_predictions_arg, axis=1)
target=np.expand_dims(target, axis=1)
result_for_excell = np.concatenate((MFCC_predictions_arg,scalo_predictions_arg,wave_predictions_arg,ensem_predictions_arg,target),1)
df = pd.DataFrame(result_for_excell,columns=['MFCC','Scalo','Wave','Ensem','target'])
count_t = time.time()
df.to_excel("/home/iichsk/workspace/wavpreprocess/all_result{}.xlsx".format(count_t), sheet_name = 'sheet1')
