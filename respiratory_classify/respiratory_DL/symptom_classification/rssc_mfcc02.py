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
num_epochs = 2000
learning_rate = 2e-3
filter_size1 = 5
filter_size2 = 3
dropout = 0.1
classes = 4
mypath = "/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/processed_audio_files_nopad/"

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]

p_id_in_file = [] # patient IDs corresponding to each file

for name in filenames:

    p_id_in_file.append(int(name.split('_')[0]))
p_id_in_file = np.array(p_id_in_file)

max_pad_len = 295 # to make the length of all MFCC equal

def extract_features(file_name):
    """
    This function takes in the path for an audio file as a string, loads it, and returns the MFCC
    of the audio"""

    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=6)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0,pad_width)), mode='constant')
        #mfccs = np.pad(mfccs, pad_width=((0,0)), mode='constant')
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccs

filepaths = [join(mypath, f) for f in filenames] # full paths of files

p_diag = pd.read_csv("/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/csv_data/data2.csv",header=None)


labels1 = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])
spt_features = []
spt_features_1 = []
def extract_image(file_name):
    im = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    #im = plt.resize(im, (640, 480))
    #print(im.shape)
    return im
# Iterate through each sound file and extract the features

for file_name in filepaths:
    data = extract_features(file_name)
    spt_features_1.append(len(data[1]))
    spt_features.append(data)



print('Finished feature extraction from ', len(spt_features), ' files')

#print('features=',features)
spt_features1 = np.array(spt_features)
#spt_features1 = np.delete(spt_features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)
#labels1 = np.delete(labels, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)

scaling_result = []
for ss in range(spt_features1.shape[1]):

    scaling_result.append(sc.fit_transform(spt_features1[:, ss, :]).reshape(spt_features1.shape[0], 1, spt_features1.shape[2]))
spt_features1 = np.concatenate(scaling_result, axis=1)


unique_elements, counts_elements = np.unique(labels1, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

le = LabelEncoder()
i_labels = le.fit_transform(labels1)
oh_labels = to_categorical(i_labels)
spt_features1 = np.reshape(spt_features1, (*spt_features1.shape, 1))
spt_train, spt_test, label_train, label_test = train_test_split(spt_features1, oh_labels, stratify=i_labels,
                                                    test_size=0.2, random_state = 42)
# df3 = pd.DataFrame(spt_test[1,:,:,0])
# count_t = time.time()
# df3.to_excel("/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/mfcc_testset_data_{}.xlsx".format(count_t), sheet_name = 'sheet1')

spt_train = np.transpose(spt_train, (0,3,1,2))
spt_test = np.transpose(spt_test, (0,3,1,2))

# construct model on cuda if available
use_cuda = torch.cuda.is_available()

# loss
criterion = nn.MultiLabelSoftMarginLoss()

class spt_cnn(nn.Module):

    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(spt_cnn, self).__init__()

        self.spt_conv1 = nn.Conv2d(1, 16, kernel_size=filter_size1)  # batchsize,16,36x291
        self.spt_pool1 = nn.MaxPool2d(2)  # 16@19*430
        self.spt_dropout1 = nn.Dropout(dropout)
        self.spt_conv2 = nn.Conv2d(16, 32, kernel_size=filter_size2)  # batchsize,32,16x143
        self.spt_pool2 = nn.MaxPool2d(2)  # 32@9*214
        self.spt_dropout2 = nn.Dropout(dropout)
        self.spt_conv3 = nn.Conv2d(32, 64, kernel_size=(1,3))  # batchsize,32,8x69
        self.spt_pool3 = nn.MaxPool2d(2)  # 64@4*106
        self.spt_dropout3 = nn.Dropout(dropout)
        self.spt_conv4 = nn.Conv2d(64, 128, kernel_size=(1,2))  # batchsize,32,4x33
        self.spt_pool4 = nn.MaxPool2d(2)  # 
        self.spt_dropout4 = nn.Dropout(dropout)
        self.spt_conv5 = nn.Conv2d(128, 256, kernel_size=(1,2))  # batchsize,32,2x15
        self.spt_pool5 = nn.MaxPool2d(2)  # 
        self.spt_dropout5 = nn.Dropout(dropout)
        self.spt_global_pool = nn.AdaptiveAvgPool2d(1)
        self.spt_fc1 = nn.Linear(256, 128)
        self.spt_bn1 = torch.nn.BatchNorm1d(128)
        self.spt_fc2 = nn.Linear(128, classes)
        self.spt_relu = nn.ReLU()
    
    def forward(self, spt_x):

        feature1 = spt_x
        spt_x = self.spt_relu(self.spt_conv1(spt_x))
        spt_x = self.spt_pool1(spt_x)  #
        spt_x = self.spt_dropout1(spt_x)
        spt_x = self.spt_relu(self.spt_conv2(spt_x))
        spt_x = self.spt_pool2(spt_x) 
        spt_x = self.spt_dropout2(spt_x)
        spt_x = self.spt_relu(self.spt_conv3(spt_x))
        spt_x = self.spt_pool3(spt_x) 
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
        spt_feature = spt_x
        spt_x = self.spt_bn1(spt_x)
        spt_feature_bn = spt_x
        spt_x = self.spt_dropout4(spt_x) 
        spt_x = self.spt_fc2(spt_x)

        return spt_x, spt_feature, spt_feature_bn

spt_cnn = spt_cnn()
spt_cnn.cuda()
# spt_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/symptom_wav_cnn_weight_4_2000_128_1631271379.8081152.pt"))

optimizer = optim.Adam(spt_cnn.parameters(), lr=learning_rate)

def train():

    z = np.random.permutation(spt_train.shape[0])

    trn_loss_list = []
    print('train start!!!!!')
    spt_cnn.train()
    
    for epoch in range(num_epochs):
        trn_loss = 0.0
        for i in range(int(spt_train.shape[0] / batch_size)):
            #spt_input = torch.Tensor(spt_train[z[i*batch_size:(i+1)* batch_size], :, :, :]).cuda()
            spt_input = torch.Tensor(spt_train[z[i*batch_size:(i+1)* batch_size], :, :, :]).cuda()

            label =  torch.Tensor(label_train[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            # grad init
            optimizer.zero_grad()
            model_output, feature_x, _ = spt_cnn(spt_input)
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
        if (epoch + 1) % 20 == 0:  #
            print("epoch: {}/{} | trn loss: {:.5f}".format(
                epoch + 1, num_epochs, trn_loss / 100
            ))
            trn_loss_list.append(trn_loss / 100)
    count_t = time.time()
    print("Model's state_dict:")
    torch.save(spt_cnn.state_dict(), "/home/iichsk/workspace/weight/symptom_wav_cnn_weight_{}_{}_{}_{}.pt".format(classes, num_epochs, batch_size, count_t))


def test(spt_test):
    spt_cnn.eval()
    test_loss = 0.0
    predictions = np.zeros((int(spt_test.shape[0]), classes))
    mfcc_features = np.zeros((int(spt_test.shape[0]), 128))
    mfcc_features_bn = np.zeros((int(spt_test.shape[0]), 128))


    correct = 0
    wrong = 0
    for j in range(int(spt_test.shape[0])):
        #spt_input = torch.Tensor(spt_test[j:(j+1), :, :, :]).cuda()
        spt_input = torch.Tensor(spt_test[j:(j+1), :, :, :]).cuda()

        test_label =  torch.Tensor(label_test[j:(j+1), :]).cuda()
        test_output, feature, feature_bn = spt_cnn(spt_input)
        feature = np.squeeze(feature)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        mfcc_features[j:j+1,:] = feature.detach().cpu().numpy()
        mfcc_features_bn[j:j+1,:] = feature_bn.detach().cpu().numpy()
        
        
    return predictions, mfcc_features, mfcc_features_bn


train()
print('train finished!!!')
_, mfcc_features_x, _ = test(spt_train)
# np.save('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/spt_feature_x_norm2',mfcc_features_x)

predictions, mfcc_features, mfcc_features_bn = test(spt_test)
# np.save('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/spt_feature_norm2',mfcc_features)

df = pd.DataFrame(mfcc_features)
count_t = time.time()
df.to_excel("/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_excell/mfcc_feature_{}.xlsx".format(count_t), sheet_name = 'sheet1')

df2 = pd.DataFrame(mfcc_features_bn)
count_t = time.time()
df.to_excel("/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_excell/mfcc_feature_bn_{}.xlsx".format(count_t), sheet_name = 'sheet1')

classpreds = np.argmax(predictions, axis=1)  # predicted classes
target = np.argmax(label_test, axis=1)  # true classes
c_names = ['normal','crackles','wheezels','Both']
#print('predictions=', classpreds)
#print('target class=', target)
# Classification Report
print(classification_report(target, classpreds, target_names=c_names))
# Confusion Matrix
print(confusion_matrix(target, classpreds))