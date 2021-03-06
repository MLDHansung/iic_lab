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

# mypath = "/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/processed_audio_files_nopad"
# mypath_waveform = "/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/waveform_image"

# filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]

# p_id_in_file = [] # patient IDs corresponding to each file
# p_id_in_file2 = [] # patient IDs corresponding to each file
# p_id_in_file_waveform = [] # patient IDs corresponding to each file

# for name in filenames:
#     p_id_in_file.append(int(name.split('_')[0]))
# p_id_in_file = np.array(p_id_in_file)

# for name in filenames:
#     p_id_in_file2.append(int(name.split('_')[0]))
# p_id_in_file2 = np.array(p_id_in_file2)

# for name in filenames:
#     p_id_in_file_waveform.append(int(name.split('_')[0]))
# p_id_in_file_waveform = np.array(p_id_in_file_waveform)
 
# def extract_features(file_name):
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

# sr=22050

# def extract_features_scalogram(file_name):
#     """
#     This function takes in the path for an audio file as a string, loads it, and returns the MFCC
#     of the audio"""
#     max_pad_len = 1393
#     try:
#         clip, sample_rate = librosa.load(file_name, sr=sr, mono=True) 
#         #print('clip',clip.shape)
#         clip_cqt = librosa.cqt(clip, hop_length=256, sr=sample_rate, fmin=30, bins_per_octave=32, n_bins=150, filter_scale=1.)    
#         clip_cqt_abs = np.abs(clip_cqt)
#         pad_width = max_pad_len - clip_cqt_abs.shape[1]
#         clip_cqt_abs = np.pad(clip_cqt_abs, pad_width=((0, 0), (0, pad_width)), mode='constant')
#         times = np.arange(len(clip))/float(sample_rate)
        
#         #clip_cqt_abs=np.log(clip_cqt_abs**2)
#     except Exception as e:
#         print("Error encountered while parsing file: ", file_name)
#         return None 
     
#     return clip_cqt_abs, clip, times

# def extract_image(file_name):
#     #file_name=
#     im = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
#     #im = plt.resize(im, (640, 480))
#     #print('file_name=',file_name)
#     return im

# filepaths = [join(mypath, f) for f in filenames] # full paths of files
# #print('filepaths',filepaths[:4])
# filepaths_waveform = [join(mypath_waveform, f)+'_waveform.png' for f in filenames] # full paths of files
# #print('filepaths_waveform',filepaths_waveform[:4])
# p_diag = pd.read_csv("/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/csv_data/data2.csv",header=None)

# labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])


# spt_features = []
# scalo_features = []
# img_features = []


# for file_name in filepaths:
#     data = extract_features(file_name)
#     #print("file_name=",file_name)
#     spt_features.append(data)

# for file_name1 in filepaths:
#     data,_,_ = extract_features_scalogram(file_name1)
#     #print("file_name=",file_name1)
#     scalo_features.append(data)

# for file_name2 in filepaths_waveform:
#     data = extract_image(file_name2)
#     #print("file_name=",file_name2)
#     img_features.append(data)

# print('Finished feature extraction from ', len(spt_features), ' files')
# print('Finished feature extraction from ', len(scalo_features), ' files')
# print('Finished feature extraction from ', len(img_features), ' files')

# spt_features = np.array(spt_features)
# spt_features1 = np.delete(spt_features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)
# labels1 = np.delete(labels, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)

# scalo_features = np.array(scalo_features)
# scalo_features1 = np.delete(scalo_features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)

# img_features = np.array(img_features)
# img_features1 = np.delete(img_features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)

# unique_elements, counts_elements = np.unique(labels1, return_counts=True)
# print(np.asarray((unique_elements, counts_elements)))

# le = LabelEncoder()
# i_labels = le.fit_transform(labels1)

# oh_labels = to_categorical(i_labels)

# spt_features1 = np.reshape(spt_features1, (*spt_features1.shape, 1))
# scalo_features1 = np.reshape(scalo_features1, (*scalo_features1.shape, 1))
# img_features1 = np.reshape(img_features1, (*img_features1.shape, 1))

# spt_train, spt_test, label_train, label_test = train_test_split(spt_features1, oh_labels, stratify=i_labels,
#                                                     test_size=0.2, random_state = 42)
# scalo_train, scalo_test, label2_train, label2_test = train_test_split(scalo_features1, oh_labels, stratify=i_labels,
#                                                     test_size=0.2, random_state = 42)
# waveform_train, waveform_test, label3_train, label3_test = train_test_split(img_features1, oh_labels, stratify=i_labels,
#                                                     test_size=0.2, random_state = 42)
# print(spt_train.shape)
# # print('spt=',spt_train[0:2,0,0,0])
# # print('scalo=',scalo_train[0:2,0,0,0])
# # print('wave=',waveform_train[0,:,:,0])
# # print('wave=',waveform_test[0,:,:,0])
# df = pd.DataFrame(waveform_test[0,:,:,0])
# df.to_excel("/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/wave_value_.xlsx", sheet_name = 'sheet1')

# spt_train = np.transpose(spt_train, (0,3,1,2))
# spt_test = np.transpose(spt_test, (0,3,1,2))
# scalo_train = np.transpose(scalo_train, (0,3,1,2))
# scalo_test = np.transpose(scalo_test, (0,3,1,2))
# waveform_train = np.transpose(waveform_train, (0,3,1,2))
# waveform_test = np.transpose(waveform_test, (0,3,1,2))

# #data loader
# '''train_loader = DataLoader(train_dataset,
#                                          batch_size=batch_size,
#                                          shuffle=True)
# val_loader = DataLoader(test_dataset,
#                                          batch_size=batch_size,
#                                          shuffle=True)
# test_loader = DataLoader(test_dataset,
#                                          batch_size=1,
#                                          shuffle=False)'''

# construct model on cuda if available
use_cuda = torch.cuda.is_available()

# loss
criterion = nn.MultiLabelSoftMarginLoss()
class spt_cnn(nn.Module):

    def __init__(self):
        # ?????? torch.nn.Module??? ???????????? ??????
        super(spt_cnn, self).__init__()

        self.spt_conv1 = nn.Conv2d(1, 16, kernel_size=5)  # 16@39*861
        self.spt_pool1 = nn.MaxPool2d(2)  # 16@19*430
        self.spt_dropout1 = nn.Dropout(spt_dropout)
        self.spt_conv2 = nn.Conv2d(16, 32, kernel_size=3)  # 32@18*429
        self.spt_pool2 = nn.MaxPool2d(2)  # 32@9*214
        self.spt_dropout2 = nn.Dropout(spt_dropout)
        self.spt_conv3 = nn.Conv2d(32, 64, kernel_size=2)  # 64@8*213
        self.spt_pool3 = nn.MaxPool2d(2)  # 64@4*106
        self.spt_dropout3 = nn.Dropout(spt_dropout)
        self.spt_conv4 = nn.Conv2d(64, 128, kernel_size=2)  # 128@3*105
        self.spt_pool4 = nn.MaxPool2d(2)  # 128@1*52
        self.spt_dropout4 = nn.Dropout(spt_dropout)
        self.spt_conv5 = nn.Conv2d(128, 128, kernel_size=2)  # 128@3*105
        self.spt_pool5 = nn.MaxPool2d(2)  # 128@1*52
        self.spt_dropout5 = nn.Dropout(spt_dropout)
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
        #spt_x = self.spt_relu(self.spt_conv5(spt_x))
        #spt_x = self.spt_pool5(spt_x)
        #spt_x = self.spt_dropout5(spt_x)
        spt_x = self.spt_global_pool(spt_x) # batchsize x netdim x 1 x 1
        spt_feature_x = spt_x
        spt_x = spt_x.view(spt_x.size(0), -1) # batchsize x netdim 
        spt_x = self.spt_fc1(spt_x) 
        spt_x = self.spt_bn1(spt_x)
        spt_x = self.spt_dropout4(spt_x)        
        spt_x = self.spt_fc2(spt_x)

        return spt_x, spt_feature_x

class scalo_cnn(nn.Module):

    def __init__(self):
        # ?????? torch.nn.Module??? ???????????? ??????
        super(scalo_cnn, self).__init__()

        self.spt_conv1 = nn.Conv2d(1, 16, kernel_size=4)  # 16@39*861
        self.spt_pool1 = nn.MaxPool2d(2)  # 16@19*430
        self.spt_dropout1 = nn.Dropout(scalo_dropout)
        self.spt_conv2 = nn.Conv2d(16, 32, kernel_size=2)  # 32@18*429
        self.spt_pool2 = nn.MaxPool2d(2)  # 32@9*214
        self.spt_dropout2 = nn.Dropout(scalo_dropout)
        self.spt_conv3 = nn.Conv2d(32, 64, kernel_size=2)  # 64@8*213
        self.spt_pool3 = nn.MaxPool2d(2)  # 64@4*106
        self.spt_dropout3 = nn.Dropout(scalo_dropout)
        self.spt_conv4 = nn.Conv2d(64, 128, kernel_size=2)  # 128@3*105
        self.spt_pool4 = nn.MaxPool2d(2)  # 128@1*52
        self.spt_dropout4 = nn.Dropout(scalo_dropout)
        self.spt_conv5 = nn.Conv2d(128, 256, kernel_size=2)  # 128@3*105
        self.spt_pool5 = nn.MaxPool2d(2)  # 128@1*52
        self.spt_dropout5 = nn.Dropout(scalo_dropout)        
        self.spt_global_pool = nn.AdaptiveAvgPool2d(1)
        self.spt_fc1 = nn.Linear(256, 128)
        self.spt_bn1 = torch.nn.BatchNorm1d(128)
        self.spt_fcdropout1 = nn.Dropout(scalo_dropout)
        self.spt_fc2 = nn.Linear(128, classes)       
        self.spt_relu = nn.ReLU()
    
    def forward(self, spt_x):
        #print(spt_x.shape)
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
        spt_x = self.spt_relu(self.spt_conv5(spt_x))
        spt_x = self.spt_pool5(spt_x)  #
        spt_x = self.spt_dropout5(spt_x)    
        spt_x = self.spt_global_pool(spt_x) # batchsize x netdim x 1 x 1
        spt_x = spt_x.view(spt_x.size(0), -1) # batchsize x netdim 
        spt_x = self.spt_fc1(spt_x) 
        spt_x = self.spt_relu(spt_x)
        spt_x = self.spt_bn1(spt_x)
        spt_feature_x = spt_x
        spt_x = self.spt_fcdropout1(spt_x)
        spt_x = self.spt_fc2(spt_x)
        #spt_x = self.spt_relu(spt_x)
        #spt_x = self.spt_bn2(spt_x)
        #spt_x = self.spt_fc3(spt_x)
        return spt_x, spt_feature_x

class img_cnn(nn.Module):

    def __init__(self):
        # ?????? torch.nn.Module??? ???????????? ??????
        super(img_cnn, self).__init__()

        self.img_conv1 = nn.Conv2d(1, 16, kernel_size=filter_size3)  # 16@39*861
        self.img_pool1 = nn.MaxPool2d(2)  # 16@19*430
        self.img_dropout1 = nn.Dropout(img_dropout)

        self.img_conv2 = nn.Conv2d(16, 32, kernel_size=filter_size4)  # 32@18*429
        self.img_pool2 = nn.MaxPool2d(2)  # 32@9*214
        self.img_dropout2 = nn.Dropout(img_dropout)

        self.img_conv3 = nn.Conv2d(32, 64, kernel_size=filter_size4)  # 64@8*213
        self.img_pool3 = nn.MaxPool2d(2)  # 64@4*106
        self.img_dropout3 = nn.Dropout(img_dropout)

        self.img_conv4 = nn.Conv2d(64, 128, kernel_size=2)  # 64@8*213
        self.img_pool4 = nn.MaxPool2d(2)  # 64@4*106
        self.img_dropout4 = nn.Dropout(img_dropout)

        self.img_conv5 = nn.Conv2d(128, 256, kernel_size=2)  # 64@8*213
        self.img_pool5 = nn.MaxPool2d(2)  # 64@4*106
        self.img_dropout5 = nn.Dropout(img_dropout)

        self.img_conv6 = nn.Conv2d(256, 512, kernel_size=2)  # 64@8*213
        self.img_pool6 = nn.MaxPool2d(2)  # 64@4*106
        self.img_dropout6 = nn.Dropout(img_dropout)

        self.img_conv7 = nn.Conv2d(512, 1024, kernel_size=2)  # 64@8*213
        self.img_pool7 = nn.MaxPool2d(2)  # 64@4*106
        self.img_dropout7 = nn.Dropout(img_dropout)

        self.img_global_pool = nn.AdaptiveAvgPool2d(1)

        self.img_fc1 = nn.Linear(1024, 512)
        self.img_bn1 = torch.nn.BatchNorm1d(512)

        self.img_fc2 = nn.Linear(512, 128)
        self.img_fc3 = nn.Linear(128, classes)

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
        img_x = self.img_dropout5(img_x)  #

        img_x = self.img_relu(self.img_conv6(img_x))
        img_x = self.img_pool6(img_x)  #
        img_x = self.img_dropout6(img_x)  #

        img_x = self.img_relu(self.img_conv7(img_x))
        #img_x = self.img_pool7(img_x)  #
        img_x = self.img_dropout7(img_x)  #

        img_x = self.img_global_pool(img_x)

        img_x = img_x.view(img_x.size(0), -1)
        

        img_x = self.img_fc1(img_x)  #
        #img_x = self.img_bn1(img_x)
        
        img_x = self.img_fc2(img_x)
        img_feature_x=img_x
        img_x = self.img_fc3(img_x)

        return img_x, img_feature_x

class concat_fc(nn.Module):
    def __init__(self):
        super(concat_fc, self).__init__()

        self.concat_conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(32,2))
        self.concat_conv1_bn = torch.nn.BatchNorm2d(64)

        self.concat_conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(32,2))
        self.concat_conv2_bn = torch.nn.BatchNorm2d(128)

        self.concat_conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(32,1))
        self.concat_conv3_bn = torch.nn.BatchNorm2d(256)
        self.concat_global_pool = nn.AdaptiveAvgPool2d(1)


        self.concat_fc1 = nn.Linear(256, 128)
        self.concat_fc1_bn = torch.nn.BatchNorm1d(128)
        self.concat_fc2 = nn.Linear(128, classes)
        self.concat_relu = nn.ReLU()
        self.concat_dropout = nn.Dropout(0.35)

    def forward(self, concat_x):   
        concat_x = self.concat_conv1(concat_x)
        concat_x = self.concat_conv1_bn(concat_x)
        concat_x = self.concat_dropout(concat_x)
        concat_x = self.concat_relu(concat_x)

        concat_x = self.concat_conv2(concat_x)
        concat_x = self.concat_conv2_bn(concat_x)
        concat_x = self.concat_dropout(concat_x)
        concat_x = self.concat_relu(concat_x)

        concat_x = self.concat_conv3(concat_x)
        concat_x = self.concat_conv3_bn(concat_x)
        concat_x = self.concat_dropout(concat_x)
        concat_x = self.concat_relu(concat_x)

        concat_x = self.concat_global_pool(concat_x)
        concat_x = concat_x.view(concat_x.size(0), -1)
        concat_x = self.concat_fc1(concat_x)
        concat_x = self.concat_fc1_bn(concat_x)
        concat_x = self.concat_dropout(concat_x)
        concat_x = self.concat_relu(concat_x)

        concat_x = self.concat_fc2(concat_x)

        return concat_x

spt_cnn = spt_cnn()
spt_cnn.cuda()
spt_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/sym_mfcc/symptom_wav_cnn_weight_4_1000_128_1626675124.4169002.pt"))
spt_optimizer = optim.Adam(spt_cnn.parameters(), lr=learning_rate)

scalo_cnn = scalo_cnn()
scalo_cnn.cuda()
scalo_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/sym_scalo/symptom_scalo_cnn_weight_4_300_128_1626674999.2049477.pt"))
scalo_optimizer = optim.Adam(scalo_cnn.parameters(), lr=learning_rate)

img_cnn = img_cnn()
img_cnn.cuda()
img_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/sym_wave/symptom_wav_cnn_weight_4_1000_128_1626951295.9009268.pt"))
img_optimizer = optim.Adam(img_cnn.parameters(), lr=learning_rate)

concat_fc = concat_fc()
concat_fc.cuda()
#concat_fc.load_state_dict(torch.load("/home/iichsk/workspace/weight/concat_fc_weight_350_128_1627272809.0676024.pt"))

concat_optimizer = optim.Adam(concat_fc.parameters(), lr=learning_rate)

def concat_train(concat_trainset,feature_label):
    num_epochs = 200
    z = np.random.permutation(concat_trainset.shape[0])
    trn_loss_list = []
    print('concat train start!!!!!')
    concat_fc.train()
    for epoch in range(num_epochs):
        trn_loss = 0.0
        for i in range(int(concat_trainset.shape[0] / batch_size)):
            concat_input = torch.Tensor(concat_trainset[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            concat_input = Variable(concat_input.data, requires_grad=True)
            label =  torch.Tensor(feature_label[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            
            # grad init
            concat_optimizer.zero_grad()
            model_output = concat_fc(concat_input)
            # calculate loss
            concat_loss = criterion(model_output, label)
            # back propagation
            concat_loss.backward(retain_graph=True)
            # weight update
            concat_optimizer.step()
            # trn_loss summary
            trn_loss += concat_loss.item()
           # del (memory issue)
            '''del loss
            del model_output'''

            # ???????????? ??????
        if (epoch + 1) % 50 == 0:  #
            print("epoch: {}/{} | trn loss: {:.8f}".format(
                epoch + 1, num_epochs, trn_loss / 100
            ))
            trn_loss_list.append(trn_loss / 100)
    count_t = time.time()
    torch.save(concat_fc.state_dict(), "/home/iichsk/workspace/weight/concat_fc_weight_{}_{}_{}.pt".format(num_epochs, batch_size, count_t))

def spt_cnn_test(spt_data):
    spt_cnn.eval()
    test_loss = 0.0
    predictions = np.zeros((int(spt_data.shape[0]), classes))
    spt_feature = np.zeros((int(spt_data.shape[0]), 128,1,1))
    correct = 0
    wrong = 0
    for j in range(int(spt_data.shape[0])):
        spt_input = torch.Tensor(spt_data[j:(j+1), :, :, :]).cuda()
        test_output, spt_feature_x = spt_cnn(spt_input)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        spt_feature[j:j+1,:] = spt_feature_x.data.cpu().numpy()

    return predictions, spt_feature

def scalo_cnn_test(scalo_data):
    scalo_cnn.eval()
    test_loss = 0.0
    predictions = np.zeros((int(scalo_data.shape[0]), classes))
    scalo_feature = np.zeros((int(scalo_data.shape[0]), 128))

    for j in range(int(scalo_data.shape[0])):
        scalo_input = torch.Tensor(scalo_data[j:(j+1), :, :, :]).cuda()
        test_output, scalo_feature_x = scalo_cnn(scalo_input)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        scalo_feature[j:j+1,:] = scalo_feature_x.data.cpu().numpy()

    return predictions, scalo_feature

def img_cnn_test(img_data):
    img_cnn.eval()
    test_loss = 0.0
    predictions = np.zeros((int(img_data.shape[0]), classes))
    img_feature = np.zeros((int(img_data.shape[0]), 128))

    for j in range(int(img_data.shape[0])):
        img_input = torch.Tensor(img_data[j:(j+1), :, :, :]).cuda()
        test_output, img_feature_x = img_cnn(img_input)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        img_feature[j:j+1,:] = img_feature_x.data.cpu().numpy()

    return predictions, img_feature

def concat_test(concat_testset):
    concat_fc.eval()
    test_loss = 0.0
    predictions = np.zeros((int(concat_testset.shape[0]), classes))
    #concat_testset=np.reshape(concat_testset, (*concat_testset.shape, 1))
    for j in range(int(concat_testset.shape[0])):
        concat_input = torch.Tensor(concat_testset[j:(j+1), :]).cuda()
        test_output = concat_fc(concat_input)

        predictions[j:j+1,:] = test_output.detach().cpu().numpy()

    return predictions


label_train = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/label_train.npy')
label_test = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/label_test.npy')

###############spectrum part###################
spt_feature_x = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/spt_feature_x.npy')
spt_feature = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/spt_feature.npy')
spt_feature_x = spt_feature_x.reshape(spt_feature_x.shape[0],spt_feature_x.shape[1])
spt_feature = spt_feature.reshape(spt_feature.shape[0],spt_feature.shape[1])

################################################

###################scalo part###################

scalo_feature_x = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/scalo_feature_x.npy')
scalo_feature = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/scalo_feature.npy')

################################################

###################waveform part###################

img_feature_x = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/img_feature_x2.npy')
img_feature = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/img_feature2.npy')
waveform_test = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/waveform_test.npy')


img_prediction, _ = img_cnn_test(waveform_test)

classpreds3 = np.argmax(img_prediction, axis=1)  # predicted classes
print('classpreds3',classpreds3.shape)

# classpreds3=np.expand_dims(classpreds3, axis=1)
target = np.argmax(label_test, axis=1)  # true classes
print('target',target.shape)
classpreds3=np.expand_dims(classpreds3, axis=1)
target=np.expand_dims(target, axis=1)
c_names = ['normal','wheezes','crackles','Both']
print('#'*10,'Waveform CNN report','#'*10)
print(classification_report(target, classpreds3, target_names=c_names))
print(confusion_matrix(target, classpreds3))
concat_pred = np.concatenate((classpreds3,target),1)


df = pd.DataFrame(concat_pred,columns=['wave','target'])
count_t = time.time()
df.to_excel("/home/iichsk/workspace/wavpreprocess/wave_predict_value_{}.xlsx".format(count_t), sheet_name = 'sheet1')

##############train################################
spt_feature_x = np.reshape(spt_feature_x, (*spt_feature_x.shape, 1))
scalo_feature_x = np.reshape(scalo_feature_x, (*scalo_feature_x.shape, 1))
img_feature_x = np.reshape(img_feature_x, (*img_feature_x.shape, 1))
concat_trainset = np.concatenate((spt_feature_x, scalo_feature_x, img_feature_x),2)
print('concat_trainset.shape',concat_trainset.shape)
concat_trainset = np.reshape(concat_trainset, (*concat_trainset.shape, 1))
concat_trainset = np.transpose(concat_trainset, (0,3,1,2))
print('concat_trainset.shape',concat_trainset.shape)
concat_train(concat_trainset,label_train)

##############test################################
spt_feature = np.reshape(spt_feature, (*spt_feature.shape, 1))
scalo_feature = np.reshape(scalo_feature, (*scalo_feature.shape, 1))
img_feature = np.reshape(img_feature, (*img_feature.shape, 1))
concat_testset = np.concatenate((spt_feature, scalo_feature, img_feature),2)
concat_testset = np.reshape(concat_testset, (*concat_testset.shape, 1))
concat_testset = np.transpose(concat_testset, (0,3,1,2))
predictions = concat_test(concat_testset)

###############report#####################################
classpreds = np.argmax(predictions, axis=1)  # predicted classes
print('classpreds', classpreds)
target = np.argmax(label_test, axis=1)  # true classes
print('target', target)
#c_names = ['Abnormal', 'Healthy']
c_names = ['normal','wheezes','crackles','Both']
# Classification Report
print('#'*10,'Triple Ensemble CNN report','#'*10)
print(classification_report(target, classpreds, target_names=c_names))
# Confusion Matrix
print(confusion_matrix(target, classpreds))



