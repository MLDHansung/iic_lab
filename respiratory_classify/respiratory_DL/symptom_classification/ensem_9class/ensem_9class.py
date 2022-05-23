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

batch_size = 40
learning_rate = 1e-3
classes = 9
MFCC_dropout = 0.3
scalo_dropout = 0.3
wave_dropout = 0.3

ensem_num_epochs = 200

# dataset load
scalo_dataset = np.load('scalo_dataset_9classes_hsk.npy')
dataset_label = np.load('dataset_label_9classes_hsk.npy')

mfcc_dataset = np.load('mfcc_dataset_9classes_khs.npy')
# dataset_label = np.load('dataset_label_9classes_khs.npy')

scalogram_trainset, scalogram_testset, train_label, test_label = train_test_split(scalo_dataset, dataset_label, stratify=dataset_label,
                                                    test_size=0.2, random_state = 42)
mfcc_trainset, mfcc_testset, train_label, test_label = train_test_split(mfcc_dataset, dataset_label, stratify=dataset_label,
                                                    test_size=0.2, random_state = 42)

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
        scalo_feature_x = scalo_x # feature size 128
        scalo_x = self.scalo_fcdropout1(scalo_x)
        scalo_x = self.scalo_fc2(scalo_x)

        return scalo_x, scalo_feature_x


class concat_fc(nn.Module):
    def __init__(self):
        super(concat_fc, self).__init__()

        self.concat_fc1 = nn.Linear(256, 128)
        self.concat_fc2 = nn.Linear(128, 64)
        self.concat_fc3 = nn.Linear(64, classes)
        self.concat_bn1 = torch.nn.BatchNorm1d(128)
        self.concat_bn2 = torch.nn.BatchNorm1d(64)

    def forward(self, MFCC_input,scalo_input):   
    
        _,MFCC_feature,_ = MFCC_cnn(MFCC_input)
        _,scalo_feature = scalo_cnn(scalo_input)
        concat_x = torch.cat((MFCC_feature, scalo_feature),1)
        concat_x = self.concat_fc1(concat_x) 
        # concat_x = self.concat_bn1(concat_x)
        concat_x = self.concat_fc2(concat_x) 
        # concat_x = self.concat_bn2(concat_x)
        concat_x = self.concat_fc3(concat_x) 

        return concat_x


MFCC_cnn = MFCC_cnn()
MFCC_cnn.cuda()
MFCC_optimizer = optim.Adam(MFCC_cnn.parameters(), lr=learning_rate)
MFCC_cnn.load_state_dict(torch.load("MFCC_cnn_9class.pt"),strict=False)

scalo_cnn = scalo_cnn()
scalo_cnn.cuda()
scalo_optimizer = optim.Adam(scalo_cnn.parameters(), lr=learning_rate)
scalo_cnn.load_state_dict(torch.load("Scalogram_cnn_9class.pt"),strict=False)

concat_fc = concat_fc()
concat_fc.cuda()
concat_optimizer = optim.Adam(concat_fc.parameters(), lr=learning_rate)
# concat_fc.load_state_dict(torch.load("ensemble_weight.pt"),strict=False)


# Ensemble train
def concat_train(mfcc_trainset,scalogram_trainset,feature_label):
    MFCC_cnn.eval()
    scalo_cnn.eval()

    z = np.random.permutation(mfcc_trainset.shape[0])
    trn_loss_list = []
    print('concat train start!!!!!')
    concat_fc.train()
    for epoch in range(ensem_num_epochs):
        trn_loss = 0.0
        for i in range(int(mfcc_trainset.shape[0] / batch_size)):
            mfcc_input = torch.Tensor(mfcc_trainset[z[i*batch_size:(i+1)* batch_size], :, :, :]).cuda()
            scalo_input = torch.Tensor(scalogram_trainset[z[i*batch_size:(i+1)* batch_size], :, :, :]).cuda()

            label_torch =  torch.Tensor(feature_label[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            # grad init
            concat_optimizer.zero_grad()
            model_output = concat_fc(mfcc_input,scalo_input)
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
    torch.save(concat_fc.state_dict(), "concat_fc_weight_{}_{}_{}.pt".format(ensem_num_epochs, batch_size, count_t))

# Ensemble test
def concat_test(mfcc_testset,scalogram_testset):
    MFCC_cnn.eval()
    scalo_cnn.eval()

    concat_fc.eval()
    test_loss = 0.0
    predictions = np.zeros((int(mfcc_testset.shape[0]), classes))

    for j in range(int(mfcc_testset.shape[0])):
        mfcc_input = torch.Tensor(mfcc_testset[j:(j+1), :, :, :]).cuda()
        scalo_input = torch.Tensor(scalogram_testset[j:(j+1), :, :, :]).cuda()

        test_output = concat_fc(mfcc_input,scalo_input)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()

    return predictions




# Ensemble part
# trainset feature concatenate and Ensemble train
concat_train(mfcc_trainset,scalogram_trainset,train_label)

# testset feature concatenate and Ensemble test
# concat_testset = np.concatenate((mfcc_testset, scalogram_testset),1)
ensem_predictions = concat_test(mfcc_testset,scalogram_testset)

# Ensemble model report
ensem_predictions_arg = np.argmax(ensem_predictions, axis=1)  
target = target = np.argmax(test_label, axis=1)
c_names = ['h_normal','d_normal_Bronchiectasis','d_normal_Bronchiolitis',
  'd_normal_COPD','d_normal_Pneumonia','d_normal_URTI','wheezes',
  'xcrackles','zBoth']
print('#'*10,'Triple Ensemble CNN1 report','#'*10)
print(classification_report(target, ensem_predictions_arg, target_names=c_names))
print(confusion_matrix(target, ensem_predictions_arg))
