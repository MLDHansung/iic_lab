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
import torch.nn.functional as F
import torch.optim as optim
import cv2
import time
from torchsummary import summary

batch_size = 128
learning_rate = 1e-3
filter_size1 = 10
filter_size2 = 2
classes = 6
spt_dropout = 0.325
img_dropout = 0.325

path = os.getcwd()
mypath = path+"/database/audio_files/"
mypath2 = path+"/database/waveform_image/"
p_diag = pd.read_csv(path+"/database/label/patient_diagnosis.csv",header=None)

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]
filenames2 = [f for f in listdir(mypath2) if (isfile(join(mypath2, f)) and f.endswith('.png'))]

p_id_in_file = [] # patient IDs corresponding to each file
p_id_in_file2 = [] # patient IDs corresponding to each file

for name in filenames:
    p_id_in_file.append(int(name[:3]))
p_id_in_file = np.array(p_id_in_file)

##### Converting sounds into images ######
def transform_mfcc(file_name):
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

def transform_image(file_name):
    im = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    return im

# Iterate through each sound file and extract the features
filepaths = [join(mypath, f) for f in filenames] # respiratory sound full paths of files
filepaths2 = [join(mypath2, f) for f in filenames2] # waveform image full paths of files
labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])

spt_features = []
img_features = []
for file_name in filepaths:
    data = transform_mfcc(file_name)
    spt_features.append(data)
    for file_name2 in filepaths2:
        if(file_name.split('/')[-1]+'.png'==file_name2.split('/')[-1]):
            data = transform_image(file_name2)
            img_features.append(data)
print('Finished feature extraction from ', len(spt_features), ' files')
print('Finished feature extraction from ', len(img_features), ' files')

# Exception Asthma, LRTI
spt_features = np.array(spt_features)
spt_features1 = np.delete(spt_features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)
img_features = np.array(img_features)
img_features1 = np.delete(img_features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)
labels1 = np.delete(labels, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)

unique_elements, counts_elements = np.unique(labels1, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

# label encoding
le = LabelEncoder()
i_labels = le.fit_transform(labels1)
oh_labels = to_categorical(i_labels)

# Add batch shape
spt_features1 = np.reshape(spt_features1, (*spt_features1.shape, 1))
img_features1 = np.reshape(img_features1, (*img_features1.shape, 1))

# Separate the dataset into a learning set and a test set 
spt_train, spt_test, label_train, label_test = train_test_split(spt_features1, oh_labels, stratify=i_labels,
                                                    test_size=0.2, random_state = 42)
img_train, img_test, label2_train, label2_test = train_test_split(img_features1, oh_labels, stratify=i_labels,
                                                    test_size=0.2, random_state = 42)
spt_train = np.transpose(spt_train, (0,3,1,2))
spt_test = np.transpose(spt_test, (0,3,1,2))
img_train = np.transpose(img_train, (0,3,1,2))
img_test = np.transpose(img_test, (0,3,1,2))
print('spt_train.shape',spt_train.shape)
print('img_train.shape',img_train.shape)
# construct model on cuda if available
use_cuda = torch.cuda.is_available()

# loss
criterion = nn.MultiLabelSoftMarginLoss()

# FC layer 제외한 MFCC CNN 
class spt_cnn(nn.Module):

    def __init__(self):
        super(spt_cnn, self).__init__()

        self.spt_conv1 = nn.Conv2d(1, 16, kernel_size=2)  # 16@39*861
        self.spt_pool1 = nn.MaxPool2d(2)  # 16@19*430
        self.spt_dropout1 = nn.Dropout(spt_dropout)
        self.spt_conv2 = nn.Conv2d(16, 32, kernel_size=2)  # 32@18*429
        self.spt_pool2 = nn.MaxPool2d(2)  # 32@9*214
        self.spt_dropout2 = nn.Dropout(spt_dropout)
        self.spt_conv3 = nn.Conv2d(32, 64, kernel_size=2)  # 64@8*213
        self.spt_pool3 = nn.MaxPool2d(2)  # 64@4*106
        self.spt_dropout3 = nn.Dropout(spt_dropout)
        self.spt_conv4 = nn.Conv2d(64, 128, kernel_size=2)  # 128@3*105
        self.spt_pool4 = nn.MaxPool2d(2)  # 128@1*52
        self.spt_dropout4 = nn.Dropout(spt_dropout)
        self.spt_global_pool = nn.AdaptiveAvgPool2d(1)
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
        
        return spt_feature_x

spt_cnn = spt_cnn()
spt_cnn.cuda()
# MFCC CNN load 중 strict=False 옵션
spt_cnn.load_state_dict(torch.load(path+"/weight/spt_cnn_weight_6_600_128_best.pt"),strict=False)
spt_optimizer = optim.Adam(spt_cnn.parameters(), lr=learning_rate)

# FC layer 제외한 Waveform CNN 
class img_cnn(nn.Module):

    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(img_cnn, self).__init__()

        self.img_conv1 = nn.Conv2d(1, 16, kernel_size=filter_size1)  # 16@39*861
        self.img_pool1 = nn.MaxPool2d(2)  # 16@19*430
        self.img_dropout1 = nn.Dropout(img_dropout)
        self.img_conv2 = nn.Conv2d(16, 32, kernel_size=filter_size2)  # 32@18*429
        self.img_pool2 = nn.MaxPool2d(2)  # 32@9*214
        self.img_dropout2 = nn.Dropout(img_dropout)
        self.img_conv3 = nn.Conv2d(32, 64, kernel_size=filter_size2)  # 64@8*213
        self.img_pool3 = nn.MaxPool2d(2)  # 64@4*106
        self.img_dropout3 = nn.Dropout(img_dropout)
        self.img_conv4 = nn.Conv2d(64, 128, kernel_size=filter_size2)  # 128@3*105
        self.img_pool4 = nn.MaxPool2d(2)  # 128@1*52
        self.img_dropout4 = nn.Dropout(img_dropout)
        self.img_conv5 = nn.Conv2d(128, 256, kernel_size=filter_size2)  # 128@3*105
        self.img_pool5 = nn.MaxPool2d(2)  # 128@1*52
        self.img_dropout5 = nn.Dropout(img_dropout)
        self.img_global_pool = nn.AdaptiveAvgPool2d(1)
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
        img_x = self.img_dropout5(img_x)  
        img_x = self.img_global_pool(img_x)
        img_feature_x = img_x

        return img_feature_x

img_cnn = img_cnn()
img_cnn.cuda()
# Waveform CNN load 중 strict=False 옵션
img_cnn.load_state_dict(torch.load(path+"/weight/img_cnn_weight_6_800_128_best.pt"),strict=False)
img_optimizer = optim.Adam(img_cnn.parameters(), lr=learning_rate)


# ensemble model
class concat_fc(nn.Module):
    def __init__(self):
        super(concat_fc, self).__init__()
        self.concat_fc1 = nn.Linear(384, 128)
        self.concat_fc2 = nn.Linear(128, classes)
        self.concat_relu = nn.ReLU()
        self.concat_bn1 = torch.nn.BatchNorm1d(128)
        self.concat_dropout1 = nn.Dropout(0.4)
           
    def forward(self, spt_x, img_x):
        spt_feature = spt_cnn(spt_x)
        img_feature = img_cnn(img_x)
        concat_x = torch.cat((spt_feature, img_feature),1)
        concat_x = concat_x.view(concat_x.size(0), -1)
        concat_x = self.concat_fc1(concat_x)  #
        concat_x = self.concat_bn1(concat_x)
        concat_x = self.concat_dropout1(concat_x)
        concat_x = self.concat_fc2(concat_x)  #
        
        return concat_x

concat_fc = concat_fc()
concat_fc.cuda()
concat_fc.load_state_dict(torch.load(path+"/weight/concat_fc_weight_450_128_best.pt"))
concat_optimizer = optim.Adam(concat_fc.parameters(), lr=learning_rate)
summary(concat_fc,[(1,40,862),(1,240,861)])

def concat_train(spt_data,img_data,feature_label):
    spt_cnn.eval() # MFCC CNN test mode
    img_cnn.eval() # Waveform CNN test mode
    num_epochs = 450
    z = np.random.permutation(spt_data.shape[0])
    trn_loss_list = []
    print('concat train start!!!!!')
    concat_fc.train()
    for epoch in range(num_epochs):
        trn_loss = 0.0
        for j in range(int(spt_data.shape[0] / batch_size)):
            spt_input = torch.Tensor(spt_data[z[j*batch_size:(j+1)*batch_size], :, :, :]).cuda()
            img_input = torch.Tensor(img_data[z[j*batch_size:(j+1)*batch_size], :, :, :]).cuda()
            label =  torch.Tensor(feature_label[z[j*batch_size:(j+1)* batch_size], :]).cuda()
            # grad init
            concat_optimizer.zero_grad()
            model_output = concat_fc(spt_input,img_input)
            # calculate loss
            concat_loss = criterion(model_output, label)
            # back propagation
            concat_loss.backward(retain_graph=True)
            # weight update
            concat_optimizer.step()
            # trn_loss summary
            trn_loss += concat_loss.item()
            # 학습과정 출력
        if (epoch + 1) % 20 == 0:  #
            print("epoch: {}/{} | trn loss: {:.7f}".format(
                epoch + 1, num_epochs, trn_loss / 100
            ))
            trn_loss_list.append(trn_loss / 100)
    count_t = time.time()
    torch.save(concat_fc.state_dict(), path+"/weight/concat_fc_weight_{}_{}_{}.pt".format(num_epochs, batch_size, count_t))

def concat_test(spt_data,img_data):
    print('test start!')
    spt_cnn.eval()
    img_cnn.eval()
    concat_fc.eval()
    test_loss = 0.0
    predictions = np.zeros((int(spt_data.shape[0]), classes))
    #target = np.zeros((int(test_feature_label.shape[0]), 6))
    correct = 0
    wrong = 0
    for j in range(int(spt_data.shape[0])):
        spt_input = torch.Tensor(spt_data[j:(j+1), :, :, :]).cuda()
        img_input = torch.Tensor(img_data[j:(j+1), :, :, :]).cuda()
        test_output = concat_fc(spt_input,img_input)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
       
    return predictions


## Ensemble Model
# concat_train(spt_train,img_train,label_train)
predictions = concat_test(spt_test,img_test)
classpreds = np.argmax(predictions, axis=1)  # predicted classes
target = np.argmax(label_test, axis=1)  # ground true classes
c_names = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Pneumonia', 'URTI', 'Healthy']
# Classification Report
print(classification_report(target, classpreds, target_names=c_names))
# Confusion Matrix
print(confusion_matrix(target, classpreds))

