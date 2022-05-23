import os
from os import listdir
from os.path import isfile, join
import librosa
import librosa.display
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchsummary import summary

batch_size = 128
num_epochs = 450
learning_rate = 1e-3
classes = 6
dropout = 0.6

path = os.getcwd()
mypath = path+"/database/audio_files/"
p_diag = pd.read_csv(path+"/database/label/patient_diagnosis.csv",header=None)

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]
filepaths = [join(mypath, f) for f in filenames] # full paths of files
p_id_in_file = [] # patient IDs corresponding to each file
for name in filenames:
    p_id_in_file.append(int(name[:3]))
p_id_in_file = np.array(p_id_in_file)
labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])

def extract_features(file_name):
    """
    This function takes in the path for an audio file as a string, loads it, and returns the MFCC
    of the audio"""
    max_pad_len = 862 # to make the length of all MFCC equal    
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=20)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccs

# Iterate through each sound file and extract the features
spt_features = []
for file_name in filepaths:
    data = extract_features(file_name)
    spt_features.append(data)
print('Finished feature extraction from ', len(spt_features), ' files')


# Exception Asthma, LRTI
spt_features = np.array(spt_features)
spt_features1 = np.delete(spt_features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)
labels1 = np.delete(labels, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)

unique_elements, counts_elements = np.unique(labels1, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

# label encoding
le = LabelEncoder()
i_labels = le.fit_transform(labels1)
oh_labels = to_categorical(i_labels)

# Add batch shape
spt_features1 = np.reshape(spt_features1, (*spt_features1.shape, 1))
# Separate the dataset into a learning set and a test set 
spt_train, spt_test, label_train, label_test = train_test_split(spt_features1, oh_labels, stratify=i_labels,
                                                    test_size=0.2, random_state = 42)
spt_train = np.transpose(spt_train, (0,3,1,2))
spt_test = np.transpose(spt_test, (0,3,1,2))

# construct model on cuda if available
use_cuda = torch.cuda.is_available()
# loss
criterion = nn.MultiLabelSoftMarginLoss()

# MFCC cnn 
class cnn(nn.Module):

    def __init__(self):
        super(cnn, self).__init__()
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
        spt_x = spt_x.view(spt_x.size(0), -1) # batchsize x netdim 

        spt_x = self.spt_fc1(spt_x) 
        spt_x2= spt_x
        spt_x = self.spt_bn1(spt_x)
        spt_x = self.spt_fc2(spt_x)  
        return spt_x
cnn = cnn()
cnn.cuda()
# backpropagation method
cnn.load_state_dict(torch.load(path+"/weight/spt_cnn_weight_6_600_128_best.pt"),strict=False)
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
summary(cnn,(1,40,862))

def train():
    z = np.random.permutation(spt_train.shape[0])
    trn_loss_list = []
    print('train start!!!!!')
    cnn.train()
    for epoch in range(num_epochs):
        trn_loss = 0.0
        for i in range(int(spt_train.shape[0] / batch_size)):
            img_input = torch.Tensor(spt_train[z[i*batch_size:(i+1)* batch_size], :, :, :]).cuda()
            label =  torch.Tensor(label_train[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            # grad init
            optimizer.zero_grad()
            model_output = cnn(img_input)
            # calculate loss
            loss = criterion(model_output, label)
            # back propagation
            loss.backward()
            # weight update
            optimizer.step()
            # trn_loss summary
            trn_loss += loss.item()

            # 학습과정 출력
        if (epoch + 1) % 20 == 0:  #
            print("epoch: {}/{} | trn loss: {:.5f}".format(
                epoch + 1, num_epochs, trn_loss / 100
            ))
            trn_loss_list.append(trn_loss / 100)
    count_t = time.time()
    print("Model's state_dict:")
    torch.save(cnn.state_dict(), path+"/weight/spt_cnn_weight_{}_{}_{}_{}.pt".format(classes, num_epochs, batch_size, count_t))


def test():
    cnn.eval()
    predictions = np.zeros((int(spt_test.shape[0]), classes))
    target = np.zeros((int(label_test.shape[0]), classes))
    
    for j in range(int(spt_test.shape[0])):
        img_input = torch.Tensor(spt_test[j:(j+1), :, :, :]).cuda()
        test_label =  torch.Tensor(label_test[j:(j+1), :]).cuda()
        test_output = cnn(img_input)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        target[j:j+1, :] = test_label.detach().cpu().numpy()

    return predictions, target

# train()
print('train finished!!!')
predictions, target = test()
classpreds = np.argmax(predictions, axis=1)  # predicted classes

target = np.argmax(target, axis=1)  # true classes
c_names = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Pneumonia', 'URTI', 'Healthy']

# Classification Report
print(classification_report(target, classpreds, target_names=c_names))
# Confusion Matrix
print(confusion_matrix(target, classpreds))