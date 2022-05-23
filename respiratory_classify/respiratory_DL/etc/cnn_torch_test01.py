from datetime import datetime
from os import listdir
from os.path import isfile, join

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
import tensorflow as tf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms


batch_size = 128
num_epochs = 350
learning_rate = 1e-3
classes = 2
mypath = "/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files2"

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]

p_id_in_file = [] # patient IDs corresponding to each file
for name in filenames:
    p_id_in_file.append(int(name[:3]))

p_id_in_file = np.array(p_id_in_file)

max_pad_len = 862 # to make the length of all MFCC equal

def extract_features(file_name):
    """
    This function takes in the path for an audio file as a string, loads it, and returns the MFCC
    of the audio"""

    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=20)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        #print('mfcc.type',type(mfccs))
        #print('mfcc.shape',mfccs.shape)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        #mfccs = np.pad(mfccs, pad_width=((0,0)), mode='constant')
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccs

filepaths = [join(mypath, f) for f in filenames] # full paths of files

p_diag = pd.read_csv("/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis2.csv",header=None)


labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])

features = []

# Iterate through each sound file and extract the features

for file_name in filepaths:
    data = extract_features(file_name)
    #print("data=",data.shape)
    features.append(data)

print('Finished feature extraction from ', len(features), ' files')
#print('features=',features)
features = np.array(features)
features1 = np.delete(features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)
labels1 = np.delete(labels, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)

#labels2 = np.delete(labels, np.where((labels != 'Healthy'))[0], axis=0)
#features2 = np.delete(features, np.where((labels != 'Healthy'))[0], axis=0)

unique_elements, counts_elements = np.unique(labels1, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

le = LabelEncoder()
i_labels = le.fit_transform(labels1)
#print('i_labels.shape=', i_labels.shape)

oh_labels = to_categorical(i_labels)
#print('oh_labels.shape=', oh_labels.shape)
features1 = np.reshape(features1, (*features1.shape, 1))
#print("features1_reshape=", features1.shape)
x_train, x_test, y_train, y_test = train_test_split(features1, oh_labels, stratify=i_labels,
                                                    test_size=0.2, random_state = 42)
x_train = np.transpose(x_train, (0,3,1,2))
x_test = np.transpose(x_test, (0,3,1,2))
#print('x_train.shape=',x_train.shape)
#print('y_test=', y_test)
x_train = torch.tensor(x_train, dtype=torch.float32).to("cuda:0")
y_train = torch.tensor(y_train, dtype=torch.int64).to("cuda:0")
x_test = torch.tensor(x_test, dtype=torch.float32).to("cuda:0")
y_test = torch.tensor(y_test, dtype=torch.int64).to("cuda:0")
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)


#data loader
train_loader = DataLoader(train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)
val_loader = DataLoader(test_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)
test_loader = DataLoader(test_dataset,
                                         batch_size=1,
                                         shuffle=False)

# construct model on cuda if available
use_cuda = torch.cuda.is_available()

# loss
criterion = nn.MultiLabelSoftMarginLoss()

class cnn(nn.Module):

    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=10)  # 16@39*861
        self.pool1 = nn.MaxPool2d(2)  # 16@19*430
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2)  # 32@18*429
        self.pool2 = nn.MaxPool2d(2)  # 32@9*214
        self.dropout2 = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2)  # 64@8*213
        self.pool3 = nn.MaxPool2d(2)  # 64@4*106
        self.dropout3 = nn.Dropout(0.2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)  # 128@3*105
        self.pool4 = nn.MaxPool2d(2)  # 128@1*52
        self.dropout4 = nn.Dropout(0.2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, classes)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        #print('conv1', x.shape)
        #print('conv1', x)
        x = self.pool1(x)  #
        #print('pool1', x.shape)
        #print('pool1', x)
        x = self.dropout1(x)
        x = self.relu(self.conv2(x))
        #print('conv2', x.shape)
        #print('conv2', x)
        x = self.pool2(x)  #
        #print('pool2', x.shape)
        #print('pool2', x)
        x = self.dropout2(x)
        x = self.relu(self.conv3(x))
        #print('conv3', x.shape)
        #print('conv3', x)
        x = self.pool3(x)  #
        #print('pool3', x.shape)
        #print('pool3', x)
        x = self.dropout3(x)  #
        x = self.relu(self.conv4(x))
        #print('conv4', x.shape)
        #print('conv4', x)
        x = self.pool4(x)  #
        #print('pool4', x.shape)
        #print('pool4', x)

        x = self.dropout4(x)  #
        x = self.global_pool(x)
        #print('global_pool.shape', x.shape)
        #print('global_pool', x)
        x = x.view(x.size(0), -1)
        #x = torch.squeeze(x)
        x = self.fc1(x)  #
        #print('fc1=', x.shape)
        #print('fc1=', x)

        return x
cnn = cnn()
cnn.cuda()
#print('model-------', cnn)
# backpropagation method
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
# hyper-parameters
num_batches = len(train_loader)

def train():
    trn_loss_list = []
    print('train start!!!!!')
    cnn.train()
    for epoch in range(num_epochs):
        trn_loss = 0.0
        for i, data in enumerate(train_loader):
            x, label = data
            print('spt_label',label)
            if use_cuda:
                x = x.cuda()
                label = label.cuda()
            # grad init
            optimizer.zero_grad()
            # forward propagation
            print('-------------------------------input.shape=',x.shape)
            #print('-------------------------------input=',x)
            model_output = cnn(x)
            #print('model_output=',model_output)
            #print('model_output=',model_output.shape)
            #print('label=',label)
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
            '''if(epoch == num_epochs-1):
                                                    feature_spt[i*] = cnn.'''
        if (epoch + 1) % 20 == 0:  #
            print("epoch: {}/{} | trn loss: {:.5f}".format(
                epoch + 1, num_epochs, trn_loss / 100
            ))
            trn_loss_list.append(trn_loss / 100)
    print("Model's state_dict:")
    for param_tensor in cnn.state_dict():
        print(param_tensor, "\t", cnn.state_dict()[param_tensor].size())



def test():
    cnn.eval()
    test_loss = 0.0
    predictions = np.zeros((int(x_test.shape[0]), classes))
    target = np.zeros((int(y_test.shape[0]), classes))
    correct = 0
    wrong = 0
    for j, test in enumerate(test_loader):
        test_x, test_label = test
        if use_cuda:
            test_x = test_x.cuda()
            test_label = test_label.cuda()
        test_output = cnn(test_x)
       

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
#print('predictions=',classpreds)
#print('target class=', target)
c_names = ['Abnormal', 'Healthy']
#c_names = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Healthy', 'Pneumonia', 'URTI']

# Classification Report
print(classification_report(target, classpreds, target_names=c_names))
# Confusion Matrix
print(confusion_matrix(target, classpreds))