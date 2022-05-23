
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
import os
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




# dataset load
mypath = "/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files2/"
p_diag = pd.read_csv("/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/csv_data/data2.csv",header=None) # patient diagnosis csv file
patient_data=pd.read_csv('/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis4.csv',names=['pid','disease'])
path='/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/'

files=[s.split('.')[0] for s in os.listdir(path) if '.txt' in s]
def getPureSample(raw_data,start,end,sr=22050):
    '''
    Takes a numpy array and spilts its using start and end args
    
    raw_data=numpy array of audio sample
    start=time
    end=time
    sr=sampling_rate
    mode=mono/stereo
    
    '''
    max_ind = len(raw_data) 
    start_ind = min(int(start * sr), max_ind)
    end_ind = min(int(end * sr), max_ind)
    return raw_data[start_ind: end_ind]

def getFilenameInfo(file):
    return file.split('_')

def slicing_data(file_name):
    files_data=[]
    for file in files:
        if str(file+'.wav') == file_name:
            data=pd.read_csv(path + file + '.txt',sep='\t',names=['start','end','crackles','wheezes'])
            name_data=getFilenameInfo(file)
            data['pid']=name_data[0]
            data['mode']=name_data[-2]
            data['filename']=file
            files_data.append(data)
    files_df=pd.concat(files_data)
    files_df.reset_index()

    patient_data.pid=patient_data.pid.astype('int32')
    files_df.pid=files_df.pid.astype('int32')

    data=pd.merge(files_df,patient_data,on='pid')

    i,c=0,0
    cnt = 0
    pureSample_list=[]
    for index,row in data.iterrows():
        maxLen=6
        start=row['start']
        end=row['end']
        filename=row['filename']
        pid=row['pid']

        audio_file_loc=mypath + filename + '.wav'
        audioArr,sampleRate=librosa.load(audio_file_loc)
        pureSample=getPureSample(audioArr,start,end,sampleRate)
        pureSample_list.append(pureSample)
     
    return pureSample_list

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
def extract_MFCC(audio):
    """
    This function takes in the path for an audio file as a string, loads it, and returns the MFCC
    of the audio"""
    max_pad_len = 862
    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccs

def extract_scalogram(clip):
    """
    This function takes in the path for an audio file as a string, loads it, and returns the scalogram
    of the audio"""
    max_pad_len = 1393
    try:
        # clip, sample_rate = librosa.load(file_name, mono=True) 
        clip_cqt = librosa.cqt(clip, hop_length=256, sr=22050, fmin=30, bins_per_octave=32, n_bins=150, filter_scale=1.)    
        clip_cqt_abs = np.abs(clip_cqt)
        pad_width = max_pad_len - clip_cqt_abs.shape[1]
        clip_cqt_abs = np.pad(clip_cqt_abs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        # times = np.arange(len(clip))/float(sample_rate)
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return clip_cqt_abs


# preprocess trainset
mfcc_trainset=[]
waveform_trainset=[]
scalogram_trainset=[]
for p_id in trainset_p_id:
    for file_name in listdir(mypath):
        if (int(file_name.split('_')[0]) == p_id):
            sliced_data_list = slicing_data(file_name)
            for ii in range(len(sliced_data_list)):
                extract_mfcc_data = extract_MFCC(sliced_data_list[ii])
                mfcc_trainset.append(extract_mfcc_data)

                extract_scalogram_data = extract_scalogram(sliced_data_list[ii])

                scalogram_trainset.append(extract_scalogram_data)
    # for file_name_wave in listdir(mypath_waveform):
    #     if (int(file_name_wave.split('_')[0]) == p_id):
    #         extract_waveform_data = cv2.imread(mypath_waveform+file_name_wave, cv2.IMREAD_GRAYSCALE)
    #         waveform_trainset.append(extract_waveform_data)

mfcc_trainset = np.array(mfcc_trainset)
scalogram_trainset = np.array(scalogram_trainset)
# waveform_trainset = np.array(waveform_trainset)
print('mfcc_trainset',mfcc_trainset.shape)
print('scalogram_trainset',scalogram_trainset.shape)

# preprocess testset
mfcc_testset=[]
waveform_testset=[]
scalogram_testset=[]
for p_id in testset_p_id:
    for file_name in listdir(mypath):
        if (int(file_name.split('_')[0]) == p_id):

            extract_mfcc_data = extract_MFCC((mypath+file_name))
            mfcc_testset.append(extract_mfcc_data)

            extract_scalogram_data = extract_scalogram((mypath+file_name))
            scalogram_testset.append(extract_scalogram_data)
    
    # for file_name_wave in listdir(mypath_waveform):
    #     if (int(file_name_wave.split('_')[0]) == p_id):
    #         extract_waveform_data = cv2.imread(mypath_waveform+file_name_wave, cv2.IMREAD_GRAYSCALE)
    #         waveform_testset.append(extract_waveform_data)

mfcc_testset = np.array(mfcc_testset)
scalogram_testset = np.array(scalogram_testset)
# waveform_testset = np.array(waveform_testset)
print('mfcc_testset',mfcc_testset.shape)
print('scalogram_testset',scalogram_testset.shape)

mfcc_trainset = np.reshape(mfcc_trainset, (*mfcc_trainset.shape, 1))
mfcc_testset = np.reshape(mfcc_testset, (*mfcc_testset.shape, 1))
scalogram_trainset = np.reshape(scalogram_trainset, (*scalogram_trainset.shape, 1))
scalogram_testset = np.reshape(scalogram_testset, (*scalogram_testset.shape, 1))
# waveform_trainset = np.reshape(waveform_trainset, (*waveform_trainset.shape, 1))
# waveform_testset = np.reshape(waveform_testset, (*waveform_testset.shape, 1))

mfcc_trainset = np.transpose(mfcc_trainset, (0,3,1,2))
mfcc_testset = np.transpose(mfcc_testset, (0,3,1,2))
scalogram_trainset = np.transpose(scalogram_trainset, (0,3,1,2))
scalogram_testset = np.transpose(scalogram_testset, (0,3,1,2))
# waveform_trainset = np.transpose(waveform_trainset, (0,3,1,2))
# waveform_testset = np.transpose(waveform_testset, (0,3,1,2))

count_t = time.time()
np.save('/home/iichsk/workspace/trainset/mfcc_trainset_{}.npy'.format(count_t),mfcc_trainset)
np.save('/home/iichsk/workspace/testset/mfcc_testset_{}.npy'.format(count_t),mfcc_testset)

# np.save('/home/iichsk/workspace/trainset/waveform_trainset_{}.npy'.format(count_t),waveform_trainset)
# np.save('/home/iichsk/workspace/testset/waveform_testset_{}.npy'.format(count_t),waveform_testset)

np.save('/home/iichsk/workspace/trainset/scalogram_trainset_{}.npy'.format(count_t),scalogram_trainset)
np.save('/home/iichsk/workspace/testset/scalogram_testset_{}.npy'.format(count_t),scalogram_testset)

np.save('/home/iichsk/workspace/trainset/train_label_{}.npy'.format(count_t),train_label)
np.save('/home/iichsk/workspace/testset/test_label_{}.npy'.format(count_t),test_label)

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
        
        return spt_feature_x

spt_cnn = spt_cnn()
spt_cnn.cuda()
# MFCC CNN load 중 strict=False 옵션
spt_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/spt_cnn_weight_6_600_128_best.pt"),strict=False)
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

        return img_feature_x

img_cnn = img_cnn()
img_cnn.cuda()
# Waveform CNN load 중 strict=False 옵션
img_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/img_cnn_weight_6_800_128_best.pt"),strict=False)
img_optimizer = optim.Adam(img_cnn.parameters(), lr=learning_rate)


# ensemble model
class concat_fc(nn.Module):
    def __init__(self):
        super(concat_fc, self).__init__()
        self.concat_fc1 = nn.Linear(384, 128)
        self.concat_fc2 = nn.Linear(128, classes)
        self.concat_relu = nn.ReLU()
        self.concat_bn1 = torch.nn.BatchNorm1d(128)
        self.concat_dropout1 = nn.Dropout(0.4)
           
    def forward(self, spt_x, img_x):
        spt_feature = spt_cnn(spt_x)
        img_feature = img_cnn(img_x)
        concat_x = torch.cat((spt_feature, img_feature),1)
        concat_x = concat_x.view(concat_x.size(0), -1)
        concat_x = self.concat_fc1(concat_x)  #
        concat_x = self.concat_bn1(concat_x)
        concat_x = self.concat_dropout1(concat_x)
        concat_x = self.concat_fc2(concat_x)  #
        
        return concat_x

concat_fc = concat_fc()
concat_fc.cuda()
concat_fc.load_state_dict(torch.load("/home/iichsk/workspace/weight/concat_fc_weight_450_128_best.pt"))
concat_optimizer = optim.Adam(concat_fc.parameters(), lr=learning_rate)

def concat_train(spt_data,img_data,feature_label):
    spt_cnn.eval() # MFCC CNN test mode
    img_cnn.eval() # Waveform CNN test mode
    num_epochs = 450
    z = np.random.permutation(spt_data.shape[0])
    trn_loss_list = []
    print('concat train start!!!!!')
    concat_fc.train()
    for epoch in range(num_epochs):
        trn_loss = 0.0
        for j in range(int(spt_data.shape[0] / batch_size)):
            spt_input = torch.Tensor(spt_data[z[j*batch_size:(j+1)*batch_size], :, :, :]).cuda()
            img_input = torch.Tensor(img_data[z[j*batch_size:(j+1)*batch_size], :, :, :]).cuda()
            label =  torch.Tensor(feature_label[z[j*batch_size:(j+1)* batch_size], :]).cuda()
            # grad init
            concat_optimizer.zero_grad()
            model_output = concat_fc(spt_input,img_input)
            # calculate loss
            concat_loss = criterion(model_output, label)
            # back propagation
            concat_loss.backward(retain_graph=True)
            # weight update
            concat_optimizer.step()
            # trn_loss summary
            trn_loss += concat_loss.item()
            # 학습과정 출력
        if (epoch + 1) % 20 == 0:  #
            print("epoch: {}/{} | trn loss: {:.7f}".format(
                epoch + 1, num_epochs, trn_loss / 100
            ))
            trn_loss_list.append(trn_loss / 100)
    count_t = time.time()
    torch.save(concat_fc.state_dict(), "/home/iichsk/workspace/weight/concat_fc_weight_{}_{}_{}.pt".format(num_epochs, batch_size, count_t))

def concat_test(spt_data,img_data):
    spt_cnn.eval()
    img_cnn.eval()
    concat_fc.eval()
    test_loss = 0.0
    predictions = np.zeros((int(spt_data.shape[0]), classes))
    #target = np.zeros((int(test_feature_label.shape[0]), 6))
    correct = 0
    wrong = 0
    for j in range(int(spt_data.shape[0])):
        spt_input = torch.Tensor(spt_data[j:(j+1), :, :, :]).cuda()
        img_input = torch.Tensor(img_data[j:(j+1), :, :, :]).cuda()
        test_output = concat_fc(spt_input,img_input)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
       
    return predictions


############################################# Ensemble Model

# concat_train(spt_train,img_train,label_train)
predictions = concat_test(spt_test,img_test)
classpreds = np.argmax(predictions, axis=1)  # predicted classes
print('classpreds', classpreds)
target = np.argmax(label_test, axis=1)  # true classes
print('target', target)
c_names = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Pneumonia', 'URTI', 'Healthy']
# Classification Report
print(classification_report(target, classpreds, target_names=c_names))
# Confusion Matrix
print(confusion_matrix(target, classpreds))

