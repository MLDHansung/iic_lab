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
mfcc_num_epochs = 2000

mfcc_trainset = np.load('/home/iichsk/workspace/trainset/mfcc_trainset.npy')
mfcc_testset = np.load('/home/iichsk/workspace/testset/mfcc_testset.npy')

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
        self.MFCC_pool1 = nn.MaxPool2d(2)  #
        self.MFCC_dropout1 = nn.Dropout(MFCC_dropout)
        self.MFCC_conv2 = nn.Conv2d(16, 32, kernel_size=3)  # 
        self.MFCC_pool2 = nn.MaxPool2d(2)  # 
        self.MFCC_dropout2 = nn.Dropout(MFCC_dropout)
        self.MFCC_conv3 = nn.Conv2d(32, 64, kernel_size=2)  # 
        self.MFCC_pool3 = nn.MaxPool2d(2)  # 
        self.MFCC_dropout3 = nn.Dropout(MFCC_dropout)
        self.MFCC_conv4 = nn.Conv2d(64, 128, kernel_size=2)  # 
        self.MFCC_pool4 = nn.MaxPool2d(2)  # 
        self.MFCC_dropout4 = nn.Dropout(MFCC_dropout)
        self.MFCC_conv5 = nn.Conv2d(128, 128, kernel_size=2)  # 
        self.MFCC_pool5 = nn.MaxPool2d(2)  # 
        self.MFCC_dropout5 = nn.Dropout(MFCC_dropout)

        self.MFCC_global_pool = nn.AdaptiveAvgPool2d(1)
        self.MFCC_bn0 = torch.nn.BatchNorm1d(128)
        self.MFCC_fc1 = nn.Linear(128, 64)
        self.MFCC_bn1 = torch.nn.BatchNorm1d(64)
        self.MFCC_fc2 = nn.Linear(64, classes)  
        self.MFCC_relu = nn.ReLU()
    
    def forward(self, MFCC_x):

        MFCC_x = self.MFCC_relu(self.MFCC_conv1(MFCC_x))
        MFCC_x = self.MFCC_pool1(MFCC_x)  #
        MFCC_x = self.MFCC_dropout1(MFCC_x)
        MFCC_x = self.MFCC_relu(self.MFCC_conv2(MFCC_x))
        MFCC_x = self.MFCC_pool2(MFCC_x)  #
        MFCC_x = self.MFCC_dropout2(MFCC_x)
        MFCC_x = self.MFCC_relu(self.MFCC_conv3(MFCC_x))
        MFCC_x = self.MFCC_pool3(MFCC_x)  #
        MFCC_x = self.MFCC_dropout3(MFCC_x)  #
        MFCC_x = self.MFCC_relu(self.MFCC_conv4(MFCC_x))
        MFCC_x = self.MFCC_pool4(MFCC_x)  #
        MFCC_x = self.MFCC_dropout4(MFCC_x)  #
        
        MFCC_x = self.MFCC_global_pool(MFCC_x) #
        MFCC_x = MFCC_x.view(MFCC_x.size(0), -1) # 
        MFCC_feature_raw = MFCC_x
        MFCC_x = self.MFCC_bn0(MFCC_x) 
        MFCC_feature_x = MFCC_x
        MFCC_x = self.MFCC_fc1(MFCC_x) 
        MFCC_x = self.MFCC_bn1(MFCC_x)
        MFCC_x = self.MFCC_dropout4(MFCC_x)        
        MFCC_x = self.MFCC_fc2(MFCC_x)

        return MFCC_x, MFCC_feature_x, MFCC_feature_raw


MFCC_cnn = MFCC_cnn()
MFCC_cnn.cuda()
MFCC_optimizer = optim.Adam(MFCC_cnn.parameters(), lr=learning_rate)
# MFCC_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/MFCC_cnn_weight_2000_128_56.pt"))

# MFCC CNN train
def MFCC_train(mfcc_trainset, label):
    z = np.random.permutation(mfcc_trainset.shape[0])
    trn_loss_list = []
    print('MFCC CNN train start!!!!!')
    MFCC_cnn.train()
    for epoch in range(mfcc_num_epochs):
        trn_loss = 0.0
        for i in range(int(mfcc_trainset.shape[0] / batch_size)):
            mfcc_input = torch.Tensor(mfcc_trainset[z[i*batch_size:(i+1)* batch_size], :, :, :]).cuda()
            label_torch =  torch.Tensor(label[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            # grad init
            MFCC_optimizer.zero_grad()
            MFCC_cnn_output, _, _ = MFCC_cnn(mfcc_input)
            # calculate loss
            loss = criterion(MFCC_cnn_output, label_torch)
            # back propagation
            loss.backward()
            # weight update
            MFCC_optimizer.step()
            # trn_loss summary
            trn_loss += loss.item()
        # 학습과정 출력
        if (epoch + 1) % 20 == 0:  #
            print("epoch: {}/{} | trn loss: {:.5f}".format(
                epoch + 1, mfcc_num_epochs, trn_loss / 100))
            trn_loss_list.append(trn_loss / 100)
    count_t = time.time()
    torch.save(MFCC_cnn.state_dict(), "/home/iichsk/workspace/weight/MFCC_cnn_weight_{}_{}_{}.pt".format(mfcc_num_epochs, batch_size, count_t))
    print("MFCC CNN Model's state_dict saved")

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

# MFCC CNN part
MFCC_train(mfcc_trainset, train_label)
_, MFCC_train_features,_ = MFCC_cnn_test(mfcc_trainset, train_label)
MFCC_predictions, MFCC_test_features, MFCC_features_raw = MFCC_cnn_test(mfcc_testset, test_label)
MFCC_test_features = np.squeeze(MFCC_test_features)
MFCC_test_features_df = pd.DataFrame(MFCC_test_features)
count_t = time.time()
MFCC_test_features_df.to_excel("/home/iichsk/workspace/respiratory_classify/result/MFCC_test_features{}.xlsx".format(count_t), sheet_name = 'sheet1')
MFCC_features_raw_df = pd.DataFrame(MFCC_features_raw)
count_t = time.time()
MFCC_features_raw_df.to_excel("/home/iichsk/workspace/respiratory_classify/result/MFCC_test_features_raw{}.xlsx".format(count_t), sheet_name = 'sheet1')

# CNN models report
MFCC_predictions_arg = np.argmax(MFCC_predictions, axis=1)
target = np.argmax(test_label, axis=1)  # true classes
c_names = ['normal','wheezes','crackles','Both']

print('#'*10,'MFCC CNN report','#'*10)
print(classification_report(target, MFCC_predictions_arg, target_names=c_names))
print(confusion_matrix(target, MFCC_predictions_arg))

# result save to excell
MFCC_predictions_arg=np.expand_dims(MFCC_predictions_arg, axis=1)
target=np.expand_dims(target, axis=1)
result_for_excell = np.concatenate((MFCC_predictions_arg,target),1)
df = pd.DataFrame(result_for_excell,columns=['MFCC','target'])
count_t = time.time()
df.to_excel("/home/iichsk/workspace/respiratory_classify/result/MFCC_result{}.xlsx".format(count_t), sheet_name = 'sheet1')
