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
from sklearn.preprocessing import MaxAbsScaler

spt_sc = MaxAbsScaler(1)
scalo_sc = MaxAbsScaler(1)
wave_sc = MaxAbsScaler(1)

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
filenames2 = [f for f in listdir(mypath_waveform) if (isfile(join(mypath_waveform, f)) and f.endswith('.png'))]

# p_id_in_file = [] # patient IDs corresponding to each file
# p_id_in_file2 = [] # patient IDs corresponding to each file
p_id_in_file_waveform = [] # patient IDs corresponding to each file

# for name in filenames:
#     p_id_in_file.append(int(name.split('_')[0]))
# p_id_in_file = np.array(p_id_in_file)

# for name in filenames:
#     p_id_in_file2.append(int(name.split('_')[0]))
# p_id_in_file2 = np.array(p_id_in_file2)

for name in filenames:
    p_id_in_file_waveform.append(int(name.split('_')[0]))
p_id_in_file_waveform = np.array(p_id_in_file_waveform)
 
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

def extract_image(file_name):
    #file_name=
    im = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    #im = plt.resize(im, (640, 480))
    #print('file_name=',file_name)
    return im

# filepaths2 = [join(mypath_waveform, f) for f in filenames2] # full paths of files
# #print('filepaths',filepaths[:4])
filepaths_waveform = [join(mypath_waveform, f)+'_waveform.png' for f in filenames] # full paths of files
# #print('filepaths_waveform',filepaths_waveform[:4])
p_diag = pd.read_csv("/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/csv_data/data2.csv",header=None)

labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file_waveform])


# spt_features = []
# scalo_features = []
img_features = []


# for file_name in filepaths:
#     data = extract_features(file_name)
#     #print("file_name=",file_name)
#     spt_features.append(data)

# for file_name1 in filepaths:
#     data,_,_ = extract_features_scalogram(file_name1)
#     #print("file_name=",file_name1)
#     scalo_features.append(data)

for file_name2 in filepaths_waveform:
    # print("file_name2",file_name)
    data = extract_image(file_name2)
    img_features.append(data)

# print('Finished feature extraction from ', len(spt_features), ' files')
# print('Finished feature extraction from ', len(scalo_features), ' files')
print('Finished feature extraction from ', len(img_features), ' files')

# spt_features = np.array(spt_features)
# spt_features1 = np.delete(spt_features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)
labels1 = np.delete(labels, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)

# scalo_features = np.array(scalo_features)
# scalo_features1 = np.delete(scalo_features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)

img_features = np.array(img_features)
img_features1 = np.delete(img_features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)

# unique_elements, counts_elements = np.unique(labels1, return_counts=True)
# print(np.asarray((unique_elements, counts_elements)))

le = LabelEncoder()
i_labels = le.fit_transform(labels1)

oh_labels = to_categorical(i_labels)

# spt_features1 = np.reshape(spt_features1, (*spt_features1.shape, 1))
# scalo_features1 = np.reshape(scalo_features1, (*scalo_features1.shape, 1))
img_features1 = np.reshape(img_features1, (*img_features1.shape, 1))

# spt_train, spt_test, label_train, label_test = train_test_split(spt_features1, oh_labels, stratify=i_labels,
#                                                     test_size=0.2, random_state = 42)
# scalo_train, scalo_test, label2_train, label2_test = train_test_split(scalo_features1, oh_labels, stratify=i_labels,
#                                                     test_size=0.2, random_state = 42)
waveform_train, waveform_test, label3_train, label3_test = train_test_split(img_features1, oh_labels, stratify=i_labels,
                                                    test_size=0.2, random_state = 42)
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
waveform_test = np.transpose(waveform_test, (0,3,1,2))

# # construct model on cuda if available
use_cuda = torch.cuda.is_available()

# loss
criterion = nn.MultiLabelSoftMarginLoss()
class spt_cnn(nn.Module):

    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
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
        # 항상 torch.nn.Module을 상속받고 시작
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
        # 항상 torch.nn.Module을 상속받고 시작
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

        self.concat_fc1 = nn.Linear(384, 128)
        self.concat_fc2 = nn.Linear(128, 64)
        self.concat_fc3 = nn.Linear(64, classes)
        self.concat_relu = nn.ReLU()
        self.concat_spt_bn1 = torch.nn.BatchNorm1d(128)
        self.concat_scalo_bn1 = torch.nn.BatchNorm1d(128)
        self.concat_img_bn1 = torch.nn.BatchNorm1d(128)
        self.concat_bn1 = torch.nn.BatchNorm1d(128)

        self.concat_bn2 = torch.nn.BatchNorm1d(64)
        self.concat_dropout1 = nn.Dropout(0.30)
        self.concat_dropout2 = nn.Dropout(0.30)

    def forward(self, spt_input, scalo_input, img_input):   
        # spt_input = sc.fit_transform(spt_input)
        spt_bn = spt_input

        # scalo_input = sc.fit_transform(scalo_input)
        scalo_bn = scalo_input

        # img_input = sc.fit_transform(img_input)
        img_bn = img_input
        # print('spt_input',spt_input.shape)
        concat_x = torch.cat((spt_input, scalo_input, img_input),1)
        # print('concat_x',concat_x.shape)
        #concat_x = concat_x.view(concat_x.size(0), -1)
        concat_x = self.concat_fc1(concat_x)  #
        #concat_x = self.concat_relu(concat_x)
        concat_x = self.concat_bn1(concat_x)
        concat_feature = concat_x
        #concat_x = self.concat_dropout1(concat_x)
        concat_x = self.concat_fc2(concat_x)  #
        #concat_x = self.concat_relu(concat_x)
        concat_x = self.concat_bn2(concat_x)
        
        #concat_x = self.concat_dropout2(concat_x)
        concat_x = self.concat_fc3(concat_x)  #

        return concat_x, concat_feature, spt_bn, scalo_bn, img_bn

spt_cnn = spt_cnn()
spt_cnn.cuda()
spt_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/symptom_wav_cnn_weight_4_2000_128_1630986830.5446396.pt"))
spt_optimizer = optim.Adam(spt_cnn.parameters(), lr=learning_rate)

scalo_cnn = scalo_cnn()
scalo_cnn.cuda()
scalo_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/sym_scalo/symptom_scalo_cnn_weight_4_300_128_1626674999.2049477.pt"))
scalo_optimizer = optim.Adam(scalo_cnn.parameters(), lr=learning_rate)

img_cnn = img_cnn()
img_cnn.cuda()
img_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/sym_wave/img_cnn_weight_1000_128_1630564413.8746724.pt"))
img_optimizer = optim.Adam(img_cnn.parameters(), lr=learning_rate)

concat_fc = concat_fc()
concat_fc.cuda()
# concat_fc.load_state_dict(torch.load("/home/iichsk/workspace/weight/concat_fc_weight_350_128_1630501918.3635547.pt"))

concat_optimizer = optim.Adam(concat_fc.parameters(), lr=learning_rate)

def concat_train(spt_feature,scalo_feature,img_feature,feature_label):
    num_epochs =  100
    z = np.random.permutation(spt_feature.shape[0])
    trn_loss_list = []
    print('concat train start!!!!!')
    concat_fc.train()
    for epoch in range(num_epochs):
        trn_loss = 0.0
        for i in range(int(spt_feature.shape[0] / batch_size)):
            spt_input = torch.Tensor(spt_feature[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            scalo_input = torch.Tensor(scalo_feature[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            img_input = torch.Tensor(img_feature[z[i*batch_size:(i+1)* batch_size], :]).cuda()


            # concat_input = Variable(concat_input.data, requires_grad=True)
            label =  torch.Tensor(feature_label[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            
            # grad init
            concat_optimizer.zero_grad()
            model_output,_ , _, _, _= concat_fc(spt_input,scalo_input,img_input)
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

            # 학습과정 출력
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

def concat_test(spt_feature, scalo_feature, img_feature):
    concat_fc.eval()
    test_loss = 0.0
    predictions = np.zeros((int(spt_feature.shape[0]), classes))
    concat_features = np.zeros((int(spt_feature.shape[0]), 128))
    mfcc_features = np.zeros((int(spt_feature.shape[0]), 128))
    scalogram_features = np.zeros((int(spt_feature.shape[0]), 128))
    waveform_features = np.zeros((int(spt_feature.shape[0]), 128))
    #concat_testset=np.reshape(concat_testset, (*concat_testset.shape, 1))
    for j in range(int(spt_feature.shape[0])):
        spt_input = torch.Tensor(spt_feature[j:(j+1), :]).cuda()
        scalo_input = torch.Tensor(scalo_feature[j:(j+1), :]).cuda()
        img_input = torch.Tensor(img_feature[j:(j+1), :]).cuda()

        test_output, concat_feature, mfcc_feature, scalogram_feature, waveform_feature = concat_fc(spt_input,scalo_input,img_input)
        # print('spt_feature.shape',spt_feature.shape)
        # print('mfcc_features.shape',mfcc_features.shape)
        
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        concat_features[j:j+1,:] = concat_feature.detach().cpu().numpy()
        mfcc_features[j:j+1,:] = mfcc_feature.detach().cpu().numpy()
        scalogram_features[j:j+1,:] = scalogram_feature.detach().cpu().numpy()
        waveform_features[j:j+1,:] = waveform_feature.detach().cpu().numpy()
        

    return predictions,concat_features,mfcc_features,scalogram_features,waveform_features


label_test = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/label_test.npy')

###############spectrum part###################
# _, spt_feature_x = spt_cnn_test(spt_train)
# print('spt_feature_x.shape=',spt_feature_x.shape)
# np.save('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/spt_feature_x',spt_feature_x)

# spt_prediction, spt_feature = spt_cnn_test(spt_test)
# np.save('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/spt_feature',spt_feature)

spt_feature_x = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/spt_feature_x_norm2.npy')
spt_feature = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/spt_feature_norm.npy')
spt_feature = spt_feature.reshape(spt_feature.shape[0],spt_feature.shape[1])
spt_feature_x = spt_feature_x.reshape(spt_feature_x.shape[0],spt_feature_x.shape[1])

# classpreds1 = np.argmax(spt_prediction, axis=1)  # predicted classes
# target = np.argmax(label_test, axis=1)  # true classes
# c_names = ['normal','crackles','wheezes','Both']# Classification Report
# print('#'*10,'MFCC CNN report','#'*10)
# print(classification_report(target, classpreds1, target_names=c_names))
# # Confusion Matrix
# print(confusion_matrix(target, classpreds1))

################################################

###################scalo part###################
# _, scalo_feature_x = scalo_cnn_test(scalo_train)
# np.save('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/scalo_feature_x',scalo_feature_x)

# scalo_prediction, scalo_feature = scalo_cnn_test(scalo_test)
# np.save('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/scalo_feature',scalo_feature)

scalo_feature_x = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/scalo_feature_x.npy')
scalo_feature = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/scalo_feature.npy')

# classpreds2 = np.argmax(scalo_prediction, axis=1)  # predicted classes
# target = np.argmax(label_test, axis=1)  # true classes
# c_names = ['normal','crackles','wheezes','Both']# Classification Report
# print('#'*10,'Scalo CNN report','#'*10)
# print(classification_report(target, classpreds2, target_names=c_names))
# Confusion Matrix
# print(confusion_matrix(target, classpreds2))

################################################

###################waveform part###################
# waveform_train = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/waveform_train.npy')
# waveform_test = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/waveform_test.npy')

# _, img_feature_x = img_cnn_test(waveform_train)
# np.save('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/img_feature_x2',img_feature_x)
img_prediction, img_feature = img_cnn_test(waveform_test)
# np.save('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/img_feature_test5',img_feature)
img_feature_x = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/img_feature_x2.npy')
img_feature = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/img_feature_test5.npy')

classpreds3 = np.argmax(img_prediction, axis=1)  # predicted classes
target = np.argmax(label_test, axis=1)  # true classes
c_names = ['normal','wheezes','crackles','Both']
print('#'*10,'Waveform CNN report','#'*10)
print(classification_report(target, classpreds3, target_names=c_names))
print(confusion_matrix(target, classpreds3))

################################################
def triple_correct(classpreds):
    pred_all = pd.read_csv("/home/iichsk/workspace/wavpreprocess/cnn_prediction_test01.csv")
    # pred_all = pd.DataFrame(pred_all, index=dates, columns=list('ABCD'))

    MFCC_pred = pred_all.MFCC
    MFCC_pred=np.array(MFCC_pred)
    Scalo_pred = pred_all['Scalo']
    Scalo_pred = np.array(Scalo_pred)
    Wave_pred = pred_all['Waveform']
    Wave_pred = np.array(Wave_pred)

    for i in range(classpreds.shape[0]):
        if (MFCC_pred[i]==Scalo_pred[i]==Wave_pred[i]):
            classpreds[i]=MFCC_pred[i]

    return classpreds

####################concat data part###################
# img_feature_x = img_feature_x.reshape(img_feature_x.shape[0],img_feature_x.shape[1])
# np.save('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/label_train',label_train)
label_train = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/label_train.npy')

# np.save('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/label_test',label_test)
label_test = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/label_test.npy')

##############train################################
# feature_bn = torch.nn.BatchNorm1d(128)

# spt_feature_x_list =[]
# scalo_feature_x_list =[]
# img_feature_x_list =[]

# for i in range (int(spt_feature_x.shape[0])):
#     print('1spt_feature_x.shape=',spt_feature_x[i:i+1,:].shape)
#     spt_feature_x_list.append(feature_bn(spt_feature_x[i:i+1,:]))
    
#     print('1scalo_feature_x.shape=',scalo_feature_x[i:i+1,:].shape)
#     scalo_feature_x_list.append(feature_bn(scalo_feature_x[i:i+1,:]))
    
#     print('1img_feature_x.shape=',img_feature_x[i:i+1,:].shape)
#     img_feature_x_list.append(feature_bn(img_feature_x[i:i+1,:]))

# spt_feature_x = np.array(spt_feature_x_list)
# scalo_feature_x = np.array(scalo_feature_x_list)
# img_feature_x = np.array(img_feature_x_list)

# print('2spt_feature_x.shape=',spt_feature_x.shape)
# print('2scalo_feature_x.shape=',scalo_feature_x.shape)
# print('2img_feature_x.shape=',img_feature_x.shape)



# spt_feature_x = np.reshape(spt_feature_x, (*spt_feature_x.shape, 1))
# scalo_feature_x = np.reshape(scalo_feature_x, (*scalo_feature_x.shape, 1))
# img_feature_x = np.reshape(img_feature_x, (*img_feature_x.shape, 1))
# concat_trainset = np.concatenate((spt_feature_x, scalo_feature_x, img_feature_x),2)
# concat_trainset = np.reshape(concat_trainset, (*concat_trainset.shape, 1))
# concat_trainset = np.transpose(concat_trainset, (0,3,1,2))

spt_feature_x = spt_sc.fit_transform(spt_feature_x)
scalo_feature_x = scalo_sc.fit_transform(scalo_feature_x)
img_feature_x = wave_sc.fit_transform(img_feature_x)

# concat_train(spt_feature_x, scalo_feature_x, img_feature_x, label_train)

##############test################################
# spt_feature_list =[]
# scalo_feature_list =[]
# img_feature_list =[]

# for i in range (int(spt_feature.shape[0])):
#     print('1spt_feature_x.shape=',spt_feature[i:i+1,:].shape)
#     spt_feature_list.append(feature_bn(spt_feature[i:i+1,:]))
    
#     print('1scalo_feature_x.shape=',scalo_feature[i:i+1,:].shape)
#     scalo_feature_list.append(feature_bn(scalo_feature[i:i+1,:]))
    
#     print('1img_feature_x.shape=',img_feature[i:i+1,:].shape)
#     img_feature_list.append(feature_bn(img_feature[i:i+1,:]))

# spt_feature = np.array(spt_feature_list)
# scalo_feature = np.array(scalo_feature_list)
# img_feature = np.array(img_feature_list)

# print('2scalo_feature.shape=',spt_feature.shape)
# print('2scalo_feature.shape=',scalo_feature.shape)
# print('2img_feature.shape=',img_feature.shape)

# spt_feature = np.reshape(spt_feature, (*spt_feature.shape, 1))
# scalo_feature = np.reshape(scalo_feature, (*scalo_feature.shape, 1))
# img_feature = np.reshape(img_feature, (*img_feature.shape, 1))
# concat_testset = np.concatenate((spt_feature, scalo_feature, img_feature),2)
# concat_testset = np.reshape(concat_testset, (*concat_testset.shape, 1))
# concat_testset = np.transpose(concat_testset, (0,3,1,2))


spt_feature = spt_sc.fit_transform(spt_feature)
scalo_feature = scalo_sc.fit_transform(scalo_feature)
img_feature = wave_sc.fit_transform(img_feature)

predictions, concat_feature, mfcc_features,scalogram_features,waveform_features = concat_test(spt_feature, scalo_feature, img_feature)
concat_feature = np.reshape(concat_feature, (*concat_feature.shape, 1))

# spt_feature = np.transpose(spt_feature, (0,2,1))
# scalo_feature = np.transpose(scalo_feature, (0,2,1))
# concat_feature = np.transpose(concat_feature, (0,2,1))
# concat_testset = np.squeeze(concat_testset)
# concat_testset = np.transpose(concat_testset, (0,2,1))
# mfcc_features = np.transpose(mfcc_features, (0,2,1))
# scalogram_features = np.transpose(scalogram_features, (0,2,1))
# waveform_features = np.transpose(waveform_features, (0,2,1))

mfcc_features = np.squeeze(mfcc_features)
scalogram_features = np.squeeze(scalogram_features)
waveform_features = np.squeeze(waveform_features)
# print(spt_feature.shape,scalo_feature.shape,concat_feature.shape)
#concat_feature = np.concatenate((spt_feature,scalo_feature,concat_feature),1)
df2 = pd.DataFrame(spt_feature)
count_t = time.time()
df2.to_excel("/home/iichsk/workspace/wavpreprocess/mfcc_feature_fc_bn_each_abs{}.xlsx".format(count_t), sheet_name = 'sheet1')

df3 = pd.DataFrame(scalo_feature)
count_t = time.time()
df3.to_excel("/home/iichsk/workspace/wavpreprocess/scalo_feature_fc_bn_each_abs{}.xlsx".format(count_t), sheet_name = 'sheet1')

df4 = pd.DataFrame(img_feature)
count_t = time.time()
df4.to_excel("/home/iichsk/workspace/wavpreprocess/waveform_feature_fc_bn_each_abs{}.xlsx".format(count_t), sheet_name = 'sheet1')




classpreds = np.argmax(predictions, axis=1)  # predicted classes

target = np.argmax(label_test, axis=1)  # true classes
print('target', target)
c_names = ['normal','wheezes','crackles','Both']
# Classification Report
print('#'*10,'Triple Ensemble CNN1 report','#'*10)
print(classification_report(target, classpreds, target_names=c_names))
# Confusion Matrix
print(confusion_matrix(target, classpreds))


classpreds_voting = triple_correct(classpreds)
c_names = ['normal','wheezes','crackles','Both']
# Classification Report
print('#'*10,'Triple Ensemble CNN2 report','#'*10)
print(classification_report(target, classpreds, target_names=c_names))
# Confusion Matrix
print(confusion_matrix(target, classpreds))

# classpreds1=np.expand_dims(classpreds1, axis=1)
# classpreds2=np.expand_dims(classpreds2, axis=1)
# classpreds3=np.expand_dims(classpreds3, axis=1)
classpreds=np.expand_dims(classpreds, axis=1)
classpreds_voting=np.expand_dims(classpreds_voting, axis=1)
target=np.expand_dims(target, axis=1)

concat_pred = np.concatenate((classpreds,classpreds_voting,target),1)
# concat_pred = np.concatenate((classpreds1, classpreds2,classpreds3, classpreds,target),1)


df = pd.DataFrame(concat_pred,columns=['T_Ensem','TV_Ensem','target'])
count_t = time.time()
df.to_excel("/home/iichsk/workspace/wavpreprocess/triple_ensem_predict_value_{}.xlsx".format(count_t), sheet_name = 'sheet1')
