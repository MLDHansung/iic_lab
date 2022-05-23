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
scalo_dropout = 0.3
scalo_num_epochs = 600


scalogram_trainset = np.load('/home/iichsk/workspace/trainset/scalogram_trainset.npy')
scalogram_testset = np.load('/home/iichsk/workspace/testset/scalogram_testset.npy')
train_label = np.load('/home/iichsk/workspace/trainset/train_label.npy')
test_label = np.load('/home/iichsk/workspace/testset/test_label.npy')

# # construct model on cuda if available
use_cuda = torch.cuda.is_available()

# loss
criterion = nn.MultiLabelSoftMarginLoss()

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

scalo_cnn = scalo_cnn()
scalo_cnn.cuda()
scalo_optimizer = optim.Adam(scalo_cnn.parameters(), lr=learning_rate)

# Scalogram CNN train
def scalo_train(scalo_trainset, label):
    z = np.random.permutation(scalo_trainset.shape[0])
    trn_loss_list = []
    print('Scalogram CNN train start!!!!!')
    scalo_cnn.train()
    
    for epoch in range(scalo_num_epochs):
        trn_loss = 0.0
        for i in range(int(scalo_trainset.shape[0] / batch_size)):
            scalo_input = torch.Tensor(scalo_trainset[z[i*batch_size:(i+1)* batch_size], :, :, :]).cuda()
            label_torch =  torch.Tensor(label[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            # grad init
            scalo_optimizer.zero_grad()
            scalo_cnn_output, _ = scalo_cnn(scalo_input)
            # calculate loss
            loss = criterion(scalo_cnn_output, label_torch)
            # back propagation
            loss.backward()
            # weight update
            scalo_optimizer.step()
            # trn_loss summary
            trn_loss += loss.item()
        # 학습과정 출력
        if (epoch + 1) % 20 == 0:  #
            print("epoch: {}/{} | trn loss: {:.5f}".format(
                epoch + 1, scalo_num_epochs, trn_loss / 100))
            trn_loss_list.append(trn_loss / 100)
    count_t = time.time()
    torch.save(scalo_cnn.state_dict(), "/home/iichsk/workspace/weight/Scalogram_cnn_weight_{}_{}_{}.pt".format(scalo_num_epochs, batch_size, count_t))
    print("Scalogram CNN Model's state_dict saved")

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

# Scalogram CNN part
scalo_train(scalogram_trainset, train_label)
_, scalo_train_features = scalo_cnn_test(scalogram_trainset, train_label)
scalo_predictions, scalo_test_features = scalo_cnn_test(scalogram_testset, test_label)
scalo_test_features_df = pd.DataFrame(scalo_test_features)
count_t = time.time()
scalo_test_features_df.to_excel("/home/iichsk/workspace/result/scalo_test_feature{}.xlsx".format(count_t), sheet_name = 'sheet1')

# Scalogram CNN models report
scalo_predictions_arg = np.argmax(scalo_predictions, axis=1) 
target = np.argmax(test_label, axis=1)  # true classes
c_names = ['normal','wheezes','crackles','Both']

print('#'*10,'Scalogram CNN report','#'*10)
print(classification_report(target, scalo_predictions_arg, target_names=c_names))
print(confusion_matrix(target, scalo_predictions_arg))


# result save to excell
scalo_predictions_arg=np.expand_dims(scalo_predictions_arg, axis=1)
target=np.expand_dims(target, axis=1)
result_for_excell = np.concatenate((scalo_predictions_arg,target),1)
df = pd.DataFrame(result_for_excell,columns=['Scalo','target'])
count_t = time.time()
df.to_excel("/home/iichsk/workspace/result/scalo_result{}.xlsx".format(count_t), sheet_name = 'sheet1')
