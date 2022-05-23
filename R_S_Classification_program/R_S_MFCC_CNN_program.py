import os
from os import listdir
from os.path import isfile, join
import librosa
import librosa.display
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchsummary import summary
batch_size = 128
learning_rate = 1e-3
classes = 4
MFCC_dropout = 0.25
mfcc_num_epochs = 100

## dataset load
path = os.getcwd()
mypath = path+"/database/audio_files/"
p_diag = pd.read_csv(path+"/database/label/patient_diagnosis_sliced.csv",header=None) #patient diagnosis

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]

p_id_in_file = [] # patient IDs corresponding to each file
for name in filenames:
    p_id_in_file.append(int(name.split('_')[0]))
p_id_in_file = np.array(p_id_in_file)

def extract_features(file_name):
    """
    This function takes in the path for an audio file as a string, loads it, and returns the MFCC
    of the audio"""
    max_pad_len = 862
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=20)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccs

filepaths = [join(mypath, f) for f in filenames] # full paths of files
labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])

spt_features = []
for file_name in filepaths:
    data = extract_features(file_name)
    spt_features.append(data)

print('Finished feature extraction from ', len(spt_features), ' files')

spt_features1 = np.array(spt_features)
le = LabelEncoder()
i_labels = le.fit_transform(labels)
oh_labels = to_categorical(i_labels)

spt_features1 = np.reshape(spt_features1, (*spt_features1.shape, 1))
spt_dataset = np.transpose(spt_features1, (0,3,1,2))
mfcc_train, mfcc_test, train_label, test_label = train_test_split(spt_dataset, oh_labels, stratify=i_labels,
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
        self.MFCC_fc1 = nn.Linear(6400, 128)
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
        # MFCC_x = self.MFCC_global_pool(MFCC_x)
        MFCC_x = MFCC_x.view(MFCC_x.size(0), -1) # 
        MFCC_x = self.MFCC_fc1(MFCC_x) 
        MFCC_feature_raw = MFCC_x
        MFCC_x = self.MFCC_fc_bn1(MFCC_x)
        MFCC_feature_x = MFCC_x
        MFCC_x = self.MFCC_fc2(MFCC_x)

        return MFCC_x, MFCC_feature_x, MFCC_feature_raw


MFCC_cnn = MFCC_cnn()
MFCC_cnn.cuda()
MFCC_optimizer = optim.Adam(MFCC_cnn.parameters(), lr=learning_rate)
# MFCC_cnn.load_state_dict(torch.load(path+"/weight/MFCC_cnn2_weight_1000_128_1631600870.018815.pt"))
summary(MFCC_cnn,(1,40,862))
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
    torch.save(MFCC_cnn.state_dict(), path+"/weight/MFCC_cnn_weight_{}_{}_{}.pt".format(mfcc_num_epochs, batch_size, count_t))
    print("MFCC CNN Model's state_dict saved")

# MFCC CNN test
def MFCC_cnn_test(mfcc_testset):
    MFCC_cnn.eval()
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
MFCC_train(mfcc_train, train_label)
MFCC_predictions, _, _ = MFCC_cnn_test(mfcc_test)
count_t = time.time()

# CNN models report
MFCC_predictions_arg = np.argmax(MFCC_predictions, axis=1)
target = np.argmax(test_label, axis=1)  # true classes
c_names = ['normal','wheezes','crackles','Both']

print('#'*10,'MFCC CNN report','#'*10)
print(classification_report(target, MFCC_predictions_arg, target_names=c_names))
print(confusion_matrix(target, MFCC_predictions_arg))
