import os
from os import listdir
from os.path import isfile, join

import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import time

os.environ["CUDA_VISIBLE_DEVICES"]="1"
batch_size = 64
learning_rate = 1e-3
classes = 4
wave_dropout = 0.3
wave_num_epochs = 2000

## dataset load
path = os.getcwd()
mypath = path+"/database/waveform_image/"
p_diag = pd.read_csv(path+"/database/label/patient_diagnosis_sliced.csv",header=None) #patient diagnosis

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.png'))]
p_id_in_file = [] # patient IDs corresponding to each file
for name in filenames:
    p_id_in_file.append(int(name.split('_')[0]))
p_id_in_file = np.array(p_id_in_file)

def extract_image(file_name):
    im = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    return im

filepaths_waveform = [join(mypath, f) for f in filenames] # full paths of files
labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])

img_features = []
for file_name2 in filepaths_waveform:
        data = extract_image(file_name2)
        img_features.append(data)

print('Finished feature extraction from ', len(img_features), ' files')

## label encoding
img_features1 = np.array(img_features)

le = LabelEncoder()
i_labels = le.fit_transform(labels)
oh_labels = to_categorical(i_labels)
img_features1 = np.reshape(img_features1, (*img_features1.shape, 1))
waveform_dataset = np.transpose(img_features1, (0,3,1,2))
waveform_train, waveform_test, train_label, test_label = train_test_split(waveform_dataset, oh_labels, stratify=i_labels,
                                                    test_size=0.2, random_state = 42)

# # construct model on cuda if available
use_cuda = torch.cuda.is_available()

# loss
criterion = nn.MultiLabelSoftMarginLoss()


class wave_cnn(nn.Module):
    def __init__(self):
        super(wave_cnn, self).__init__()
        self.wave_conv1 = nn.Conv2d(1, 16, kernel_size=(2,10))  
        self.wave_pool1 = nn.MaxPool2d(2)  
        self.wave_dropout1 = nn.Dropout(wave_dropout)
        self.wave_conv2 = nn.Conv2d(16, 32, kernel_size=(2,10))  
        self.wave_pool2 = nn.MaxPool2d(2) 
        self.wave_dropout2 = nn.Dropout(wave_dropout)
        self.wave_conv3 = nn.Conv2d(32, 64, kernel_size=(2,10))  
        self.wave_pool3 = nn.MaxPool2d(2)  
        self.wave_dropout3 = nn.Dropout(wave_dropout)
        self.wave_conv4 = nn.Conv2d(64, 128, kernel_size=2)  
        self.wave_pool4 = nn.MaxPool2d(2)  
        self.wave_dropout4 = nn.Dropout(wave_dropout)
        self.wave_conv5 = nn.Conv2d(128, 256, kernel_size=2)  
        self.wave_pool5 = nn.MaxPool2d(2)  
        self.wave_dropout5 = nn.Dropout(wave_dropout)
        self.wave_conv6 = nn.Conv2d(256, 512, kernel_size=2) 
        self.wave_pool6 = nn.MaxPool2d(2)  
        self.wave_dropout6 = nn.Dropout(wave_dropout)
        self.wave_global_pool = nn.AdaptiveAvgPool2d(1)
        self.wave_fc1 = nn.Linear(512, 256)
        self.wave_fc2 = nn.Linear(256, 128)
        self.wave_bn2 = torch.nn.BatchNorm1d(128)
        self.wave_fc3 = nn.Linear(128, classes)
        self.wave_relu = nn.ReLU()
       
    
    def forward(self, wave_x):
        wave_x = self.wave_relu(self.wave_conv1(wave_x))
        wave_x = self.wave_pool1(wave_x) 
        wave_x = self.wave_dropout1(wave_x)
        wave_x = self.wave_relu(self.wave_conv2(wave_x))
        wave_x = self.wave_pool2(wave_x) 
        wave_x = self.wave_dropout2(wave_x)
        wave_x = self.wave_relu(self.wave_conv3(wave_x))
        wave_x = self.wave_pool3(wave_x) 
        wave_x = self.wave_dropout3(wave_x) 
        wave_x = self.wave_relu(self.wave_conv4(wave_x))
        wave_x = self.wave_pool4(wave_x) 
        wave_x = self.wave_dropout4(wave_x) 
        wave_x = self.wave_relu(self.wave_conv5(wave_x))
        wave_x = self.wave_pool5(wave_x) 
        wave_x = self.wave_dropout5(wave_x) 
        wave_x = self.wave_relu(self.wave_conv6(wave_x))
        wave_x = self.wave_pool6(wave_x) 
        wave_x = self.wave_dropout6(wave_x) 
        
        wave_x = self.wave_global_pool(wave_x)
        wave_x = wave_x.view(wave_x.size(0), -1)
        wave_x = self.wave_fc1(wave_x) 
        wave_x = self.wave_fc2(wave_x)
        wave_feature_x = self.wave_bn2(wave_x) # feature size 128
        wave_x = self.wave_fc3(wave_x)

        return wave_x, wave_feature_x

wave_cnn = wave_cnn()
wave_cnn.cuda()
wave_optimizer = optim.Adam(wave_cnn.parameters(), lr=learning_rate)
wave_cnn.load_state_dict(torch.load(path+"/weight/Waveform_cnn_weight_1000_128_1631515583.3403041.pt"),strict=False)

# wave CNN train
def wave_cnn_train(wave_trainset, label):
    z = np.random.permutation(wave_trainset.shape[0])
    trn_loss_list = []
    print('wave CNN train start!!!!!')
    wave_cnn.train()
    for epoch in range(wave_num_epochs):
        trn_loss = 0.0
        for i in range(int(wave_trainset.shape[0] / batch_size)):
            wave_input = torch.Tensor(wave_trainset[z[i*batch_size:(i+1)* batch_size], :, :, :]).cuda()
            label_torch =  torch.Tensor(label[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            # grad init
            wave_optimizer.zero_grad()
            wave_cnn_output,_ = wave_cnn(wave_input)
            # calculate loss
            loss = criterion(wave_cnn_output, label_torch)
            # back propagation
            loss.backward()
            # weight update
            wave_optimizer.step()
            # trn_loss summary
            trn_loss += loss.item()
        # 학습과정 출력
        if (epoch + 1) % 20 == 0:  #
            print("epoch: {}/{} | trn loss: {:.5f}".format(
                epoch + 1, wave_num_epochs, trn_loss / 100))
            trn_loss_list.append(trn_loss / 100)
    count_t = time.time()
    torch.save(wave_cnn.state_dict(), path+"/weight/waveform_cnn_weight_{}_{}_{}.pt".format(wave_num_epochs, batch_size, count_t))
    print("wave CNN Model's state_dict saved")

# wave CNN test
def wave_cnn_test(wave_testset):
    wave_cnn.eval()
    predictions = np.zeros((int(wave_testset.shape[0]), classes))

    for j in range(int(wave_testset.shape[0])):
        wave_input = torch.Tensor(wave_testset[j:(j+1), :, :, :]).cuda()
        test_output, _= wave_cnn(wave_input)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()

    return predictions

# Waveform CNN part
# wave_cnn_train(waveform_train, train_label)
wave_predictions = wave_cnn_test(waveform_test)

## wave CNN model report
target = np.argmax(test_label, axis=1)  # true classes
c_names = ['normal','wheezes','crackles','Both']
wave_predictions_arg = np.argmax(wave_predictions, axis=1)  
print('#'*10,'Waveform CNN report','#'*10)
print(classification_report(target, wave_predictions_arg, target_names=c_names))
print(confusion_matrix(target, wave_predictions_arg))
