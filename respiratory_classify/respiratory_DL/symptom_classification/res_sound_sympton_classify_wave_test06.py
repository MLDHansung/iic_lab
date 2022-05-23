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

batch_size = 128
num_epochs = 2000
learning_rate = 1e-3
filter_size3 = (4,10)
filter_size4 = (2, 10)
img_dropout = 0.35
classes = 4
mypath = "/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/waveform_640_240/"

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.png'))]

p_id_in_file = [] # patient IDs corresponding to each file

for name in filenames:

    p_id_in_file.append(int(name.split('_')[0]))
p_id_in_file = np.array(p_id_in_file)

max_pad_len = 259 # to make the length of all MFCC equal

def extract_features(file_name):
    """
    This function takes in the path for an audio file as a string, loads it, and returns the MFCC
    of the audio"""

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

filepaths = [join(mypath, f) for f in filenames] # full paths of files

p_diag = pd.read_csv("/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/csv_data/data2.csv",header=None)


labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])
print(labels)
def extract_image(file_name):
    im = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    return im

# Iterate through each sound file and extract the features
img_features = []
for file_name in filepaths:
    data = extract_image(file_name)
    img_features.append(data)
print('max',max(spt_features_1))
print('min',min(spt_features_1))

print('Finished feature extraction from ', len(img_features), ' files')

#print('features=',features)
img_features = np.array(img_features)
img_features1 = np.delete(img_features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)
labels1 = np.delete(labels, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)

unique_elements, counts_elements = np.unique(labels1, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

le = LabelEncoder()
i_labels = le.fit_transform(labels1)
oh_labels = to_categorical(i_labels)
img_features1 = np.reshape(img_features1, (*img_features1.shape, 1))
img_train, img_test, label_train, label_test = train_test_split(img_features1, oh_labels, stratify=i_labels,
                                                    test_size=0.2, random_state = 42)

img_train = np.transpose(img_train, (0,3,1,2))
img_test = np.transpose(img_test, (0,3,1,2))

# construct model on cuda if available
use_cuda = torch.cuda.is_available()

# loss
criterion = nn.MultiLabelSoftMarginLoss()

'''class spt_cnn(nn.Module):

    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(spt_cnn, self).__init__()

        self.spt_conv1 = nn.Conv2d(1, 16, kernel_size=2)  # 16@39*861
        self.spt_pool1 = nn.MaxPool2d(2)  # 16@19*430
        self.spt_dropout1 = nn.Dropout(dropout)
        self.spt_conv2 = nn.Conv2d(16, 32, kernel_size=2)  # 32@18*429
        self.spt_pool2 = nn.MaxPool2d(2)  # 32@9*214
        self.spt_dropout2 = nn.Dropout(dropout)
        self.spt_conv3 = nn.Conv2d(32, 64, kernel_size=2)  # 64@8*213
        self.spt_pool3 = nn.MaxPool2d(2)  # 64@4*106
        self.spt_dropout3 = nn.Dropout(dropout)
        self.spt_conv4 = nn.Conv2d(64, 128, kernel_size=2)  # 128@3*105
        self.spt_pool4 = nn.MaxPool2d(2)  # 128@1*52
        self.spt_dropout4 = nn.Dropout(dropout)
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
        spt_x = self.spt_global_pool(spt_x) # batchsize x netdim x 1 x 1
        spt_feature_x = spt_x
        spt_x = spt_x.view(spt_x.size(0), -1) # batchsize x netdim 

        spt_x = self.spt_fc1(spt_x) 
        spt_x = self.spt_bn1(spt_x)
        spt_x = self.spt_fc2(spt_x)

        return spt_x, spt_feature_x

spt_cnn = spt_cnn()
spt_cnn.cuda()
optimizer = optim.Adam(spt_cnn.parameters(), lr=learning_rate)'''

class cnn(nn.Module):

    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(cnn, self).__init__()

        self.img_conv1 = nn.Conv2d(1, 16, kernel_size=filter_size3)  # 16@39*861
        self.img_pool1 = nn.MaxPool2d(2)  # 16@19*430
        self.img_dropout1 = nn.Dropout(img_dropout)

        self.img_conv2 = nn.Conv2d(16, 32, kernel_size=filter_size4)  # 32@18*429
        self.img_pool2 = nn.MaxPool2d(2)  # 32@9*214
        self.img_dropout2 = nn.Dropout(img_dropout)

        self.img_conv3 = nn.Conv2d(32, 64, kernel_size=filter_size4)  # 64@8*213
        self.img_pool3 = nn.MaxPool2d(2)  # 64@4*106
        self.img_dropout3 = nn.Dropout(img_dropout)

        self.img_conv4 = nn.Conv2d(64, 128, kernel_size=filter_size4)  # 64@8*213
        self.img_pool4 = nn.MaxPool2d(2)  # 64@4*106
        self.img_dropout4 = nn.Dropout(img_dropout)

        self.img_global_pool = nn.AdaptiveAvgPool2d(1)

        self.img_fc1 = nn.Linear(128, 64)
        self.img_bn1 = torch.nn.BatchNorm1d(64)

        self.img_fc2 = nn.Linear(64, classes)

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

        img_x = self.img_global_pool(img_x)

        img_x = img_x.view(img_x.size(0), -1)
        img_feature_x=img_x

        img_x = self.img_fc1(img_x)  #
        img_x = self.img_bn1(img_x)

        img_x = self.img_fc2(img_x)


        

        return img_x, img_feature_x
img_cnn = cnn()
img_cnn.cuda()
#print('model-------', cnn)
# backpropagation method
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
        if (epoch + 1) % 100 == 0:  #
            print("epoch: {}/{} | trn loss: {:.5f}".format(
                epoch + 1, num_epochs, trn_loss / 100
            ))
            trn_loss_list.append(trn_loss / 100)
    count_t = time.time()
    print("Model's state_dict:")
    torch.save(img_cnn.state_dict(), "/home/iichsk/workspace/weight/sym_classify_wave/symptom_wav_cnn_weight_{}_{}_{}_{}.pt".format(classes, num_epochs, batch_size, count_t))


def test():
    img_cnn.eval()
    test_loss = 0.0
    predictions = np.zeros((int(img_test.shape[0]), classes))
    target = np.zeros((int(label_test.shape[0]), classes))
    correct = 0
    wrong = 0
    for j in range(int(img_test.shape[0])):
        #spt_input = torch.Tensor(spt_test[j:(j+1), :, :, :]).cuda()
        img_input = torch.Tensor(img_test[j:(j+1), :, :, :]).cuda()

        test_label =  torch.Tensor(label_test[j:(j+1), :]).cuda()
        test_output, _ = img_cnn(img_input)

        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        target[j:j+1, :] = test_label.detach().cpu().numpy()
        pred = test_output.data.max(1)[1]
        one_hot_pred = torch.zeros(1, classes)
        one_hot_pred[range(one_hot_pred.shape[0]), pred]=1
        correct += one_hot_pred[0, 0].eq(test_label.data[0, 0].cpu()).cpu().sum()
        wrong +=  len(test_label)-one_hot_pred[0, 0].eq(test_label.data[0, 0].cpu()).cpu().sum()


    print(" Error: {}/{} ({:.2f}%)".format(wrong,wrong+correct, 100.*(float(wrong)/float(correct+wrong))))
    #print("eval_prediction.shape:", predictions.shape)
    return predictions, target

train()
print('train finished!!!')
predictions, target = test()
classpreds = np.argmax(predictions, axis=1)  # predicted classes
target = np.argmax(target, axis=1)  # true classes
c_names = ['normal','crackles','wheezes','Both']
#print('predictions=', classpreds)
#print('target class=', target)
# Classification Report
print(classification_report(target, classpreds, target_names=c_names))
# Confusion Matrix
print(confusion_matrix(target, classpreds))