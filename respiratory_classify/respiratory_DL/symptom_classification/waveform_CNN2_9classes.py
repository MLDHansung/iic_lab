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
import time
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()

batch_size = 128
num_epochs = 200
learning_rate = 1e-3
filter_size3 = (2,10)
filter_size4 = (2, 10)
img_dropout = 0.3
classes = 9

waveform_dataset = np.load('waveform_dataset_9classes_dell.npy')
dataset_label = np.load('dataset_label_9classes_dell.npy')
print('dataset_label=',dataset_label.shape)
img_train, img_test, label_train, label_test = train_test_split(waveform_dataset, dataset_label, stratify=dataset_label,
                                                    test_size=0.2, random_state = 42)

print('img_train',img_train.shape)
# img_train = np.transpose(img_train, (0,3,1,2))
# img_test = np.transpose(img_test, (0,3,1,2))


use_cuda = torch.cuda.is_available()

# loss
criterion = nn.MultiLabelSoftMarginLoss()


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
img_cnn = img_cnn()
img_cnn.cuda()
#print('model-------', cnn)
# backpropagation method
# img_cnn.load_state_dict(torch.load("/home/iichsk/workspace/respiratory_classify/weight/sym_classify_wave/symptom_wav_cnn_weight_4_1000_128_1630997427.7513587.pt"))

optimizer = optim.Adam(img_cnn.parameters(), lr=learning_rate)
# hyper-parameters

def train():

    z = np.random.permutation(img_train.shape[0])

    trn_loss_list = []
    print('train start!!!!!')
    img_cnn.train()
    for epoch in range(num_epochs):
        trn_loss = 0.0
        for i in range(int(img_train.shape[0] / batch_size)):
            #spt_input = torch.Tensor(spt_train[z[i*batch_size:(i+1)* batch_size], :, :, :]).cuda()
            img_input = torch.Tensor(img_train[z[i*batch_size:(i+1)* batch_size], :, :, :]).cuda()

            label =  torch.Tensor(label_train[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            # grad init
            optimizer.zero_grad()
            model_output, _ = img_cnn(img_input)
            # calculate loss
            loss = criterion(model_output, label)
            # back propagation
            loss.backward()
            # weight update
            optimizer.step()

            # trn_loss summary
            trn_loss += loss.item()
           # del (memory issue)
            '''del loss
            del model_output'''

            # 학습과정 출력
        if (epoch + 1) % 10 == 0:  #
            print("epoch: {}/{} | trn loss: {:.5f}".format(
                epoch + 1, num_epochs, trn_loss / 100
            ))
            trn_loss_list.append(trn_loss / 100)
    count_t = time.time()
    print("Model's state_dict:")
    torch.save(img_cnn.state_dict(), "/home/iichsk/workspace/weight/symptom_wav_cnn_weight_{}_{}_{}_{}.pt".format(classes, num_epochs, batch_size, count_t))


def test():
    img_cnn.eval()
    test_loss = 0.0
    predictions = np.zeros((int(img_test.shape[0]), classes))
    img_features = np.zeros((int(img_test.shape[0]), 128))
    correct = 0
    wrong = 0
    for j in range(int(img_test.shape[0])):
        #spt_input = torch.Tensor(spt_test[j:(j+1), :, :, :]).cuda()
        img_input = torch.Tensor(img_test[j:(j+1), :, :, :]).cuda()

        test_label =  torch.Tensor(label_test[j:(j+1), :]).cuda()
        test_output, img_feature = img_cnn(img_input)

        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        img_features[j:j+1,:] = img_feature.data.detach().cpu().numpy()
        
    return predictions, img_features

train()
print('train finished!!!')
predictions, img_feature = test()

img_feature = np.squeeze(img_feature)

df2 = pd.DataFrame(img_feature)
count_t = time.time()
df2.to_excel("/home/iichsk/workspace/respiratory_classify/img_feature_{}.xlsx".format(count_t), sheet_name = 'sheet1')

classpreds = np.argmax(predictions, axis=1)  # predicted classes
target = np.argmax(label_test, axis=1)  # true classes
c_names = ['h_normal','d_normal_Bronchiectasis','d_normal_Bronchiolitis',
  'd_normal_COPD','d_normal_Pneumonia','d_normal_URTI','wheezes',
  'xcrackles','zBoth']

#print('target class=', target)
# Classification Report
print(classification_report(target, classpreds, target_names=c_names))
# Confusion Matrix
print(confusion_matrix(target, classpreds))