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
scalo_dropout = 0.3
scalo_num_epochs = 600

path = os.getcwd()
mypath = path+"/database/audio_files/"
p_diag = pd.read_csv(path+"/database/label/patient_diagnosis_sliced.csv",header=None)

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]

p_id_in_file = [] # patient IDs corresponding to each file
for name in filenames:
    p_id_in_file.append(int(name.split('_')[0]))
p_id_in_file = np.array(p_id_in_file)

sr=22050
def extract_features_scalogram(file_name):
    """
    This function takes in the path for an audio file as a string, loads it, and returns the MFCC
    of the audio"""
    max_pad_len = 1393
    try:
        clip, sample_rate = librosa.load(file_name, sr=sr, mono=True) 
        clip_cqt = librosa.cqt(clip, hop_length=256, sr=sample_rate, fmin=30, bins_per_octave=32, n_bins=150, filter_scale=1.)    
        clip_cqt_abs = np.abs(clip_cqt)
        pad_width = max_pad_len - clip_cqt_abs.shape[1]
        clip_cqt_abs = np.pad(clip_cqt_abs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        times = np.arange(len(clip))/float(sample_rate)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return clip_cqt_abs, clip, times

filepaths = [join(mypath, f) for f in filenames] # full paths of files
labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])

scalo_features = []
for file_name in filepaths:
    data,_,_ = extract_features_scalogram(file_name)
    scalo_features.append(data)

print('Finished feature extraction from ', len(scalo_features), ' files')

le = LabelEncoder()
i_labels = le.fit_transform(labels)
oh_labels = to_categorical(i_labels)

scalo_features1 = np.array(scalo_features)
scalo_features1 = np.reshape(scalo_features1, (*scalo_features1.shape, 1))
scalo_dataset = np.transpose(scalo_features1, (0,3,1,2))
scalo_train, scalo_test, train_label, test_label = train_test_split(scalo_dataset, oh_labels, stratify=i_labels,
                                                    test_size=0.2, random_state = 42)
print('scalo_train.shape',scalo_train.shape)
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
scalo_cnn.load_state_dict(torch.load(path+"/weight/Scalogram_cnn_weight_1000_128_1631521741.4954169.pt"),strict=False)
summary(scalo_cnn,(1,150,1393))

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
    torch.save(scalo_cnn.state_dict(), path+"/weight/Scalogram_cnn_weight_{}_{}_{}.pt".format(scalo_num_epochs, batch_size, count_t))
    print("Scalogram CNN Model's state_dict saved")

# Scalogram CNN test
def scalo_cnn_test(scalogram_testset):
    scalo_cnn.eval()
    predictions = np.zeros((int(scalogram_testset.shape[0]), classes))
    scalo_features = np.zeros((int(scalogram_testset.shape[0]), 128))

    for j in range(int(scalogram_testset.shape[0])):
        scalo_input = torch.Tensor(scalogram_testset[j:(j+1), :, :, :]).cuda()
        test_output, scalo_feature = scalo_cnn(scalo_input)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        scalo_features[j:j+1,:] = scalo_feature.data.cpu().numpy()

    return predictions, scalo_features

# Scalogram CNN part
# scalo_train(scalo_train, train_label)
scalo_predictions, _ = scalo_cnn_test(scalo_test)
count_t = time.time()

# Scalogram CNN models report
scalo_predictions_arg = np.argmax(scalo_predictions, axis=1) 
target = np.argmax(test_label, axis=1)  # true classes
c_names = ['normal','wheezes','crackles','Both']

print('#'*10,'Scalogram CNN report','#'*10)
print(classification_report(target, scalo_predictions_arg, target_names=c_names))
print(confusion_matrix(target, scalo_predictions_arg))

