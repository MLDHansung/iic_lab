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
from scipy import interpolate

batch_size = 128
num_epochs =300
learning_rate = 1e-3
filter_size1 = (8,2)
filter_size2 = 2
dropout = 0.3
classes = 4
mypath = "/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/processed_audio_files_nopad/"

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]

p_id_in_file = [] # patient IDs corresponding to each file

for name in filenames:

    p_id_in_file.append(int(name.split('_')[0]))
p_id_in_file = np.array(p_id_in_file)


min_level_db= -100
def normalize_mel(S):
    return np.clip((S-min_level_db)/-min_level_db,0,1)


def extract_features_melspectrogram(file_name):
    y = librosa.load(file_name,16000)[0]
    S =  librosa.feature.melspectrogram(y=y, n_mels=80, n_fft=512, win_length=400, hop_length=160) # 320/80
    norm_log_S = normalize_mel(librosa.power_to_db(S, ref=np.max))
    x = np.linspace(0, 697, S.shape[1])
    y = np.linspace(0, 40, S.shape[0])
    #pad_width = max_pad_len - mfccs.shape[1]
    #mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    #mfccs = np.pad(mfccs, pad_width=((0,0)), mode='constant')
    f = interpolate.interp2d(x, y, norm_log_S, kind='linear')
    x_new = np.arange(0, 697)
    y_new = np.arange(0, 40)
    norm_log_S = f(x_new, y_new)

    return norm_log_S
                        
sr=22050
def extract_features_scalogram(file_name):
    """
    This function takes in the path for an audio file as a string, loads it, and returns the MFCC
    of the audio"""
   
    try:
        clip, sample_rate = librosa.load(file_name, sr=sr, mono=True) 
        #print('clip',clip.shape)
        clip_cqt = librosa.cqt(clip, hop_length=256, sr=sample_rate, fmin=30, bins_per_octave=32, n_bins=150, filter_scale=1.)
        clip_cqt_abs = np.abs(clip_cqt)
        times = np.arange(len(clip))/float(sample_rate)
        #clip_cqt_abs=np.log(clip_cqt_abs**2)
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return clip_cqt_abs, clip, times

max_pad_len = 259 # to make the length of all MFCC equal
def extract_features(file_name):
    """
    This function takes in the path for an audio file as a string, loads it, and returns the MFCC
    of the audio"""

    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=20)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        #pad_width = max_pad_len - mfccs.shape[1]
        #mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        #mfccs = np.pad(mfccs, pad_width=((0,0)), mode='constant')
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccs, audio

filepaths = [join(mypath, f) for f in filenames] # full paths of files

p_diag = pd.read_csv("/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/csv_data/data2.csv",header=None)


labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])
spt_features = []

def extract_image(file_name):
    im = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    #im = plt.resize(im, (640, 480))
    #print(im.shape)
    return im
# Iterate through each sound file and extract the features
cnt = 0
for file_name in filepaths:
    _, data = extract_features(file_name)
    cnt += 1
    #data = extract_features_melspectrogram(file_name)
    spt_features.append(data)
    if cnt % 1000 == 0:
        print("data no.{} feature extrating... ".format(cnt))


print('Finished feature extraction from ', len(spt_features), ' files')

spt_features = np.array(spt_features)
spt_features1 = np.delete(spt_features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)
labels1 = np.delete(labels, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)

unique_elements, counts_elements = np.unique(labels1, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

le = LabelEncoder()
i_labels = le.fit_transform(labels1)
oh_labels = to_categorical(i_labels)
spt_features1 = np.reshape(spt_features1, (*spt_features1.shape, 1))

spt_train, spt_test, label_train, label_test = train_test_split(spt_features1, oh_labels, stratify=i_labels,
                                                    test_size=0.2, random_state = 42)

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

        self.spt_conv1 = nn.Conv2d(1, 32, kernel_size=filter_size1)  # 16@39*861
        self.spt_pool1 = nn.MaxPool2d(2)  # 16@19*430
        self.spt_dropout1 = nn.Dropout(dropout)
        self.spt_conv2 = nn.Conv2d(32, 64, kernel_size=filter_size2)  # 32@18*429
        self.spt_pool2 = nn.MaxPool2d(2)  # 32@9*214
        self.spt_dropout2 = nn.Dropout(dropout)
        self.spt_conv3 = nn.Conv2d(64, 128, kernel_size=filter_size2)  # 64@8*213
        self.spt_pool3 = nn.MaxPool2d(2)  # 64@4*106
        self.spt_dropout3 = nn.Dropout(dropout)
        self.spt_conv4 = nn.Conv2d(128, 256, kernel_size=filter_size2)  # 128@3*105
        self.spt_pool4 = nn.MaxPool2d(2)  # 128@1*52
        self.spt_dropout4 = nn.Dropout(dropout)
        self.spt_conv5 = nn.Conv2d(256, 512, kernel_size=filter_size2)  # 128@3*105
        self.spt_pool5 = nn.MaxPool2d(2)  # 128@1*52
        self.spt_dropout5 = nn.Dropout(dropout)
        
        self.spt_global_pool = nn.AdaptiveAvgPool2d(1)
        self.spt_fc1 = nn.Linear(256, 128)
        self.spt_bn1 = torch.nn.BatchNorm1d(128)
        self.spt_fcdropout1 = nn.Dropout(dropout)

        self.spt_fc2 = nn.Linear(128, 64) 
        self.spt_bn2= torch.nn.BatchNorm1d(64)
        self.spt_fc3 = nn.Linear(64, classes) 
        #self.spt_bn3= torch.nn.BatchNorm1d(16)
        #self.spt_fc4 = nn.Linear(16, classes) 
        self.spt_relu = nn.ReLU()
    
    def forward(self, spt_x):
        #print(spt_x.shape)
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
        #spt_x = self.spt_relu(self.spt_conv5(spt_x))
        #spt_x = self.spt_pool5(spt_x)  #
        #spt_x = self.spt_dropout5(spt_x)
        
        spt_x = self.spt_global_pool(spt_x) # batchsize x netdim x 1 x 1
        spt_feature_x = spt_x
        spt_x = spt_x.view(spt_x.size(0), -1) # batchsize x netdim 

        spt_x = self.spt_fc1(spt_x) 
        spt_x = self.spt_relu(spt_x)
        spt_x = self.spt_bn1(spt_x)
        spt_x = self.spt_fcdropout1(spt_x)
        spt_x = self.spt_fc2(spt_x)
        spt_x = self.spt_relu(spt_x)

        #spt_x = self.spt_bn2(spt_x)
        spt_x = self.spt_fc3(spt_x)
        return spt_x, spt_feature_x

spt_cnn = spt_cnn()
spt_cnn.cuda()
optimizer = optim.Adam(spt_cnn.parameters(), lr=learning_rate)

'''class cnn(nn.Module):

    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(cnn, self).__init__()

        self.img_conv1 = nn.Conv2d(1, 16, kernel_size=filter_size1)  # 16@39*861
        self.img_pool1 = nn.MaxPool2d(2)  # 16@19*430
        self.img_dropout1 = nn.Dropout(dropout)
        self.img_conv2 = nn.Conv2d(16, 32, kernel_size=filter_size2)  # 32@18*429
        self.img_pool2 = nn.MaxPool2d(2)  # 32@9*214
        self.img_dropout2 = nn.Dropout(dropout)
        self.img_conv3 = nn.Conv2d(32, 64, kernel_size=filter_size2)  # 64@8*213
        self.img_pool3 = nn.MaxPool2d(2)  # 64@4*106
        self.img_dropout3 = nn.Dropout(dropout)
        self.img_conv4 = nn.Conv2d(64, 128, kernel_size=filter_size2)  # 128@3*105
        self.img_pool4 = nn.MaxPool2d(2)  # 128@1*52
        self.img_dropout4 = nn.Dropout(dropout)
        self.img_conv5 = nn.Conv2d(128, 256, kernel_size=filter_size2)  # 128@3*105
        self.img_pool5 = nn.MaxPool2d(2)  # 128@1*52
        self.img_dropout5 = nn.Dropout(dropout)
        self.img_global_pool = nn.AdaptiveAvgPool2d(1)
        self.img_fc1 = nn.Linear(256, 128)
        self.img_bn1 = torch.nn.BatchNorm1d(128)
        self.img_fc2 = nn.Linear(128, classes)
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
        img_x = img_x.view(img_x.size(0), -1)
        img_x = self.img_fc1(img_x)  #
        img_x = self.img_fc2(img_x)
  #

        return img_x
cnn = cnn()
cnn.cuda()
#print('model-------', cnn)
# backpropagation method
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
# hyper-parameters'''

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
            model_output, _ = spt_cnn(spt_input)
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
        if (epoch + 1) % 50 == 0:  #
            print("epoch: {}/{} | trn loss: {:.5f}".format(
                epoch + 1, num_epochs, trn_loss / 100
            ))
            trn_loss_list.append(trn_loss / 100)
    count_t = time.time()
    print("Model's state_dict:")
    torch.save(spt_cnn.state_dict(), "/home/iichsk/workspace/weight/sym_classify_mfcc/symptom_spt_cnn_weight_{}_{}_{}_{}.pt".format(classes, num_epochs, batch_size, count_t))


def test():
    spt_cnn.eval()
    test_loss = 0.0
    predictions = np.zeros((int(spt_test.shape[0]), classes))
    target = np.zeros((int(label_test.shape[0]), classes))
    correct = 0
    wrong = 0
    for j in range(int(spt_test.shape[0])):
        #spt_input = torch.Tensor(spt_test[j:(j+1), :, :, :]).cuda()
        spt_input = torch.Tensor(spt_test[j:(j+1), :, :, :]).cuda()

        test_label =  torch.Tensor(label_test[j:(j+1), :]).cuda()
        test_output, _ = spt_cnn(spt_input)

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
print('target class=', target)
# Classification Report
print(classification_report(target, classpreds, target_names=c_names))
# Confusion Matrix
print(confusion_matrix(target, classpreds))

spt1_classpreds=np.expand_dims(spt1_classpreds, axis=1)
spt2_classpreds=np.expand_dims(spt2_classpreds, axis=1)
concat_classpreds=np.expand_dims(concat_classpreds, axis=1)
target=np.expand_dims(target, axis=1)

print('spt1_classpreds.shape',spt1_classpreds.shape)
print('target.shape',target.shape)
concat_pred = np.concatenate((spt1_classpreds, spt2_classpreds, concat_classpreds,target),1)

'''df = pd.DataFrame(concat_pred,columns=['spt1','spt2','concat','target'])
print(df)
count_t = time.time()
df.to_excel("/home/iichsk/workspace/wavpreprocess/predict_value_{}.xlsx".format(count_t), sheet_name = 'sheet1')
'''