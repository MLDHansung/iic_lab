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
mypath = "/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files2/"
mypath_waveform = "/home/iichsk/workspace/dataset/full_image/audio_and_txt_files2/"
p_diag = pd.read_csv("/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv",header=None)

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]

max_pad_len = 862 # to make the length of all MFCC equal
def extract_MFCC(file_name):
    """
    This function takes in the path for an audio file as a string, loads it, and returns the MFCC
    of the audio"""

    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=20)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccs


cnt=0
mfcc_testset=[]
waveform_testset=[]
p_id_in_file = [] # patient IDs corresponding to each file
p_name=[]
for name in filenames:
    p_id_in_file.append(int(name.split('_')[0]))
    p_name.append(name)
    extract_mfcc_data = extract_MFCC((mypath+name))
    mfcc_testset.append(extract_mfcc_data)
    cnt+=1
    if(cnt%100==0):
        print('MFCC extracting no.{}'.format(cnt))
    extract_waveform_data = cv2.imread(mypath_waveform+name+'.png', cv2.IMREAD_GRAYSCALE)
    waveform_testset.append(extract_waveform_data)
p_name = np.array(p_name)

# p_name_df = pd.DataFrame(p_name)
# count_t = time.time()
# p_name_df.to_excel("/home/iichsk/workspace/wavpreprocess/p_name{}.xlsx".format(count_t), sheet_name = 'sheet1')

p_id_in_file = np.array(p_id_in_file)
dataset_p_id = p_id_in_file
mfcc_testset = np.array(mfcc_testset)
waveform_testset = np.array(waveform_testset)

labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])

print('Finished feature extraction from ', len(mfcc_testset), ' files')
unique_elements, counts_elements = np.unique(labels, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

dataset_p_id = np.array(dataset_p_id)
mfcc_testset = np.delete(mfcc_testset, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)
waveform_testset = np.delete(waveform_testset, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)
labels = np.delete(labels, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)

le = LabelEncoder()
i_labels = le.fit_transform(labels)
oh_labels = to_categorical(i_labels)

trainset_p_id, testset_p_id, train_label, test_label = train_test_split(dataset_p_id, oh_labels, stratify=i_labels,
                                                    test_size=0.2, random_state = 42)
testset_p_id = np.array(testset_p_id)
test_label = np.array(test_label)
target = np.argmax(test_label, axis=1)

testset_p_id=np.expand_dims(testset_p_id, axis=1)
target=np.expand_dims(target, axis=1)

print('testset_p_id',testset_p_id.shape)
print('target',target.shape)
testset_p_id_label = np.concatenate((testset_p_id, target),1)
testset_p_id_df = pd.DataFrame(testset_p_id_label)
count_t = time.time()
testset_p_id_df.to_excel("/home/iichsk/workspace/wavpreprocess/testset_p_id_{}.xlsx".format(count_t), sheet_name = 'sheet1')


# # mfcc_trainset = np.reshape(mfcc_trainset, (*mfcc_trainset.shape, 1))
# mfcc_testset = np.reshape(mfcc_testset, (*mfcc_testset.shape, 1))
# # waveform_trainset = np.reshape(waveform_trainset, (*waveform_trainset.shape, 1))
# waveform_testset = np.reshape(waveform_testset, (*waveform_testset.shape, 1))

# # mfcc_trainset = np.transpose(mfcc_trainset, (0,3,1,2))
# mfcc_testset = np.transpose(mfcc_testset, (0,3,1,2))
# # waveform_trainset = np.transpose(waveform_trainset, (0,3,1,2))
# waveform_testset = np.transpose(waveform_testset, (0,3,1,2))

# count_t = time.time()
# # np.save('/home/iichsk/workspace/trainset/mfcc_trainset_{}.npy'.format(count_t),mfcc_trainset)
# # np.save('/home/iichsk/workspace/testset/mfcc_testset_{}.npy'.format(count_t),mfcc_testset)
# # np.save('/home/iichsk/workspace/trainset/waveform_trainset_{}.npy'.format(count_t),waveform_trainset)
# # np.save('/home/iichsk/workspace/testset/waveform_testset_{}.npy'.format(count_t),waveform_testset)
# # np.save('/home/iichsk/workspace/trainset/train_label_{}.npy'.format(count_t),train_label)
# # np.save('/home/iichsk/workspace/testset/test_label_{}.npy'.format(count_t),test_label)


# # construct model on cuda if available
# use_cuda = torch.cuda.is_available()

# # loss
# criterion = nn.MultiLabelSoftMarginLoss()
# class spt_cnn(nn.Module):

#     def __init__(self):
#         # 항상 torch.nn.Module을 상속받고 시작
#         super(spt_cnn, self).__init__()

#         self.spt_conv1 = nn.Conv2d(1, 16, kernel_size=2)  # 16@39*861
#         self.spt_pool1 = nn.MaxPool2d(2)  # 16@19*430
#         self.spt_dropout1 = nn.Dropout(spt_dropout)
#         self.spt_conv2 = nn.Conv2d(16, 32, kernel_size=2)  # 32@18*429
#         self.spt_pool2 = nn.MaxPool2d(2)  # 32@9*214
#         self.spt_dropout2 = nn.Dropout(spt_dropout)
#         self.spt_conv3 = nn.Conv2d(32, 64, kernel_size=2)  # 64@8*213
#         self.spt_pool3 = nn.MaxPool2d(2)  # 64@4*106
#         self.spt_dropout3 = nn.Dropout(spt_dropout)
#         self.spt_conv4 = nn.Conv2d(64, 128, kernel_size=2)  # 128@3*105
#         self.spt_pool4 = nn.MaxPool2d(2)  # 128@1*52
#         self.spt_dropout4 = nn.Dropout(spt_dropout)
#         self.spt_global_pool = nn.AdaptiveAvgPool2d(1)
#         self.spt_fc1 = nn.Linear(128, 64)
#         self.spt_bn1 = torch.nn.BatchNorm1d(64)
#         self.spt_fc2 = nn.Linear(64, classes)  
#         self.spt_relu = nn.ReLU()
    
#     def forward(self, spt_x):

#         spt_x = self.spt_relu(self.spt_conv1(spt_x))
#         spt_x = self.spt_pool1(spt_x)  #
#         spt_x = self.spt_dropout1(spt_x)
#         spt_x = self.spt_relu(self.spt_conv2(spt_x))
#         spt_x = self.spt_pool2(spt_x)  #
#         spt_x = self.spt_dropout2(spt_x)
#         spt_x = self.spt_relu(self.spt_conv3(spt_x))
#         spt_x = self.spt_pool3(spt_x)  #
#         spt_x = self.spt_dropout3(spt_x)  #
#         spt_x = self.spt_relu(self.spt_conv4(spt_x))
#         spt_x = self.spt_pool4(spt_x)  #
#         spt_x = self.spt_dropout4(spt_x)  #
#         spt_x = self.spt_global_pool(spt_x) # batchsize x netdim x 1 x 1
#         spt_feature_x = spt_x
#         spt_x = spt_x.view(spt_x.size(0), -1) # batchsize x netdim 

#         spt_x = self.spt_fc1(spt_x) 
#         spt_x = self.spt_bn1(spt_x)
#         spt_x = self.spt_fc2(spt_x)

#         return spt_x, spt_feature_x

# spt_cnn = spt_cnn()
# spt_cnn.cuda()
# spt_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/spt_cnn_weight_6_600_128_best.pt"))
# spt_optimizer = optim.Adam(spt_cnn.parameters(), lr=learning_rate)

# class img_cnn(nn.Module):

#     def __init__(self):
#         # 항상 torch.nn.Module을 상속받고 시작
#         super(img_cnn, self).__init__()

#         self.img_conv1 = nn.Conv2d(1, 16, kernel_size=filter_size1)  # 16@39*861
#         self.img_pool1 = nn.MaxPool2d(2)  # 16@19*430
#         self.img_dropout1 = nn.Dropout(img_dropout)
#         self.img_conv2 = nn.Conv2d(16, 32, kernel_size=filter_size2)  # 32@18*429
#         self.img_pool2 = nn.MaxPool2d(2)  # 32@9*214
#         self.img_dropout2 = nn.Dropout(img_dropout)
#         self.img_conv3 = nn.Conv2d(32, 64, kernel_size=filter_size2)  # 64@8*213
#         self.img_pool3 = nn.MaxPool2d(2)  # 64@4*106
#         self.img_dropout3 = nn.Dropout(img_dropout)
#         self.img_conv4 = nn.Conv2d(64, 128, kernel_size=filter_size2)  # 128@3*105
#         self.img_pool4 = nn.MaxPool2d(2)  # 128@1*52
#         self.img_dropout4 = nn.Dropout(img_dropout)
#         self.img_conv5 = nn.Conv2d(128, 256, kernel_size=filter_size2)  # 128@3*105
#         self.img_pool5 = nn.MaxPool2d(2)  # 128@1*52
#         self.img_dropout5 = nn.Dropout(img_dropout)
#         self.img_global_pool = nn.AdaptiveAvgPool2d(1)
#         self.img_fc1 = nn.Linear(256, 128)
#         self.img_fc2 = nn.Linear(128, classes)
        

#         self.img_relu = nn.ReLU()
       
    
#     def forward(self, img_x):
        
#         img_x = self.img_relu(self.img_conv1(img_x))
#         img_x = self.img_pool1(img_x)  #
#         img_x = self.img_dropout1(img_x)
#         img_x = self.img_relu(self.img_conv2(img_x))
#         img_x = self.img_pool2(img_x)  #
#         img_x = self.img_dropout2(img_x)
#         img_x = self.img_relu(self.img_conv3(img_x))
#         img_x = self.img_pool3(img_x)  #
#         img_x = self.img_dropout3(img_x)  #
#         img_x = self.img_relu(self.img_conv4(img_x))
#         img_x = self.img_pool4(img_x)  #
#         img_x = self.img_dropout4(img_x)  #
#         img_x = self.img_relu(self.img_conv5(img_x))
#         img_x = self.img_pool5(img_x)  #
#         img_x = self.img_dropout5(img_x)  
#         img_x = self.img_global_pool(img_x)
#         img_feature_x = img_x
#         img_x = img_x.view(img_x.size(0), -1)
#         img_x = self.img_fc1(img_x)  #
#         img_x = self.img_fc2(img_x)  #
#         return img_x, img_feature_x

# img_cnn = img_cnn()
# img_cnn.cuda()
# img_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/img_cnn_weight_6_800_128_best.pt"))
# img_optimizer = optim.Adam(img_cnn.parameters(), lr=learning_rate)


# # hyper-parameters
# class concat_fc(nn.Module):
#     def __init__(self):
#         super(concat_fc, self).__init__()
       
#         self.concat_fc1 = nn.Linear(384, 128)
#         self.concat_fc2 = nn.Linear(128, classes)
#         self.concat_relu = nn.ReLU()
#         self.concat_bn1 = torch.nn.BatchNorm1d(128)
#         self.concat_dropout1 = nn.Dropout(0.4)
        
#     def forward(self, concat_x):

#         concat_x = concat_x.view(concat_x.size(0), -1)
#         concat_x = self.concat_fc1(concat_x)  #
#         concat_x = self.concat_bn1(concat_x)
#         concat_x = self.concat_dropout1(concat_x)
#         concat_x = self.concat_fc2(concat_x)  #
#         return concat_x


# concat_fc = concat_fc()
# concat_fc.cuda()
# #print('model-------', cnn)
# # backpropagation method
# concat_fc.load_state_dict(torch.load("/home/iichsk/workspace/weight/concat_fc_weight_450_128_best.pt"))

# concat_optimizer = optim.Adam(concat_fc.parameters(), lr=learning_rate)
# z = np.random.permutation(mfcc_testset.shape[0])

# def concat_train(concat_trainset,feature_label):
#     num_epochs = 450
#     z = np.random.permutation(concat_trainset.shape[0])

#     trn_loss_list = []
#     print('concat train start!!!!!')
#     concat_fc.train()
#     for epoch in range(num_epochs):
#         trn_loss = 0.0
#         for i in range(int(concat_trainset.shape[0] / batch_size)):
#             concat_input = torch.Tensor(concat_trainset[z[i*batch_size:(i+1)* batch_size], :, :, :]).cuda()
#             concat_input = Variable(concat_input.data, requires_grad=True)
#             label =  torch.Tensor(feature_label[z[i*batch_size:(i+1)* batch_size], :]).cuda()
#             # grad init
#             concat_optimizer.zero_grad()
#             model_output = concat_fc(concat_input)
#             # calculate loss
#             concat_loss = criterion(model_output, label)
#             # back propagation
#             concat_loss.backward(retain_graph=True)
#             # weight update
#             concat_optimizer.step()
#             # trn_loss summary
#             trn_loss += concat_loss.item()
#            # del (memory issue)
#             '''del loss
#             del model_output'''

#             # 학습과정 출력
#         if (epoch + 1) % 20 == 0:  #
#             print("epoch: {}/{} | trn loss: {:.5f}".format(
#                 epoch + 1, num_epochs, trn_loss / 100
#             ))
#             trn_loss_list.append(trn_loss / 100)
#     count_t = time.time()
#     torch.save(concat_fc.state_dict(), "/home/iichsk/workspace/weight/concat_fc_weight_{}_{}_{}.pt".format(num_epochs, batch_size, count_t))

# def spt_cnn_test(spt_data):
#     spt_cnn.eval()
#     test_loss = 0.0
#     predictions = np.zeros((int(spt_data.shape[0]), classes))
#     spt_feature = np.zeros((int(spt_data.shape[0]), 128, 1,1))
#     for j in range(int(spt_data.shape[0])):
#         spt_input = torch.Tensor(spt_data[j:(j+1), :, :, :]).cuda()
#         test_output, spt_feature_x = spt_cnn(spt_input)
#         predictions[j:j+1,:] = test_output.detach().cpu().numpy()
#         spt_feature[j:j+1,:,:,:] = spt_feature_x.data.cpu().numpy()
#     return predictions, spt_feature

# def img_cnn_test(img_data):
#     img_cnn.eval()
#     test_loss = 0.0
#     predictions = np.zeros((int(img_data.shape[0]), classes))
#     img_feature = np.zeros((int(img_data.shape[0]), 256, 1,1))
#     for j in range(int(img_data.shape[0])):
#         img_input = torch.Tensor(img_data[j:(j+1), :, :, :]).cuda()
#         test_output, img_feature_x = img_cnn(img_input)
#         predictions[j:j+1,:] = test_output.detach().cpu().numpy()
#         img_feature[j:j+1,:,:,:] = img_feature_x.data.cpu().numpy()

#     return predictions, img_feature

# def concat_test(concat_testset):
#     concat_fc.eval()
#     test_loss = 0.0
#     predictions = np.zeros((int(concat_testset.shape[0]), classes))

#     for j in range(int(concat_testset.shape[0])):
#         concat_input = torch.Tensor(concat_testset[j:(j+1), :, :, :]).cuda()
#         test_output = concat_fc(concat_input)
#         predictions[j:j+1,:] = test_output.detach().cpu().numpy()

#     return predictions


# #spectrum part
# spt_prediction, spt_feature = spt_cnn_test(mfcc_testset)

# MFCC_predictions_arg = np.argmax(spt_prediction, axis=1)  # predicted classes
# target = np.argmax(oh_labels, axis=1)  # true classes
# c_names = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Pneumonia', 'URTI', 'Healthy']
# # Classification Report
# print(classification_report(target, MFCC_predictions_arg, target_names=c_names))
# # Confusion Matrix
# print(confusion_matrix(target, MFCC_predictions_arg))


# #waveform part
# #_, img_feature_x = img_cnn_test(img_train)
# img_prediction, img_feature = img_cnn_test(waveform_testset)
# wave_predictions_arg = np.argmax(img_prediction, axis=1)  # predicted classes
# # Classification Report
# print(classification_report(target, wave_predictions_arg, target_names=c_names))
# # Confusion Matrix
# print(confusion_matrix(target, wave_predictions_arg))

# concat_testset = np.concatenate((spt_feature, img_feature),1)
# ensem_predictions = concat_test(concat_testset)
# ensem_predictions_arg = np.argmax(ensem_predictions, axis=1)  # predicted classes


# # Classification Report
# print(classification_report(target, ensem_predictions_arg, target_names=c_names))
# # Confusion Matrix
# print(confusion_matrix(target, ensem_predictions_arg))

# # result save to excell
# MFCC_predictions_arg=np.expand_dims(MFCC_predictions_arg, axis=1)
# wave_predictions_arg=np.expand_dims(wave_predictions_arg, axis=1)
# ensem_predictions_arg=np.expand_dims(ensem_predictions_arg, axis=1)
# target=np.expand_dims(target, axis=1)
# result_for_excell = np.concatenate((MFCC_predictions_arg,wave_predictions_arg,ensem_predictions_arg,target),1)
# df = pd.DataFrame(result_for_excell,columns=['MFCC','Wave','Ensem','target'])
# count_t = time.time()
# df.to_excel("/home/iichsk/workspace/wavpreprocess/disease_all_result{}.xlsx".format(count_t), sheet_name = 'sheet1')

