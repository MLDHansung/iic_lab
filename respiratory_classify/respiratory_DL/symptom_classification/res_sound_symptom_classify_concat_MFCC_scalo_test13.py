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
from torch.autograd import Variable
import time


batch_size = 128
learning_rate = 1e-3
filter_size1 = 5
filter_size2 = 3
filter_size3 = (4,10)
filter_size4 = (2, 10)
classes = 4
spt_dropout = 0.3
img_dropout = 0.3
mypath = "/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/processed_audio_files_nopad"
#mypath2 = "/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/processed_audio_files_nopad"

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]
#filenames2 = [f for f in listdir(mypath2) if (isfile(join(mypath2, f)) and f.endswith('.png'))]

p_id_in_file = [] # patient IDs corresponding to each file
p_id_in_file2 = [] # patient IDs corresponding to each file

for name in filenames:
    p_id_in_file.append(int(name.split('_')[0]))
p_id_in_file = np.array(p_id_in_file)

for name in filenames:
    p_id_in_file2.append(int(name.split('_')[0]))
p_id_in_file2 = np.array(p_id_in_file2)

 # to make the length of all MFCC equal

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
        #mfccs = np.pad(mfccs, pad_width=((0,0)), mode='constant')
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccs

sr=22050
def extract_features_scalogram(file_name):
    """
    This function takes in the path for an audio file as a string, loads it, and returns the MFCC
    of the audio"""

    max_pad_len = 1393
    try:
        clip, sample_rate = librosa.load(file_name, sr=sr, mono=True) 
        #print('clip',clip.shape)
        clip_cqt = librosa.cqt(clip, hop_length=256, sr=sample_rate, fmin=30, bins_per_octave=32, n_bins=150, filter_scale=1.)    
        clip_cqt_abs = np.abs(clip_cqt)
        pad_width = max_pad_len - clip_cqt_abs.shape[1]
        clip_cqt_abs = np.pad(clip_cqt_abs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        times = np.arange(len(clip))/float(sample_rate)
        
        #clip_cqt_abs=np.log(clip_cqt_abs**2)
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return clip_cqt_abs, clip, times

filepaths = [join(mypath, f) for f in filenames] # full paths of files
#filepaths2 = [join(mypath2, f) for f in filenames2] # full paths of files

p_diag = pd.read_csv("/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/csv_data/data2.csv",header=None)


labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])
labels2 = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file2])


spt_features = []
img_features = []

def extract_image(file_name):
    im = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    #im = plt.resize(im, (640, 480))
    #print(im.shape)
    return im
# Iterate through each sound file and extract the features

for file_name in filepaths:
    data = extract_features(file_name)
    #print("data=",data.shape)
    spt_features.append(data)

for file_name1 in filepaths:
    data,_,_ = extract_features_scalogram(file_name1)
    #print("data=",data.shape)
    img_features.append(data)

print('Finished feature extraction from ', len(spt_features), ' files')
print('Finished feature extraction from ', len(img_features), ' files')

#print('features=',features)
spt_features = np.array(spt_features)
spt_features1 = np.delete(spt_features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)
labels1 = np.delete(labels, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)

img_features = np.array(img_features)
#print('img_features', img_features.shape)
img_features1 = np.delete(img_features, np.where((labels2 == 'Asthma') | (labels2 == 'LRTI'))[0], axis=0)
img_labels1 = np.delete(labels2, np.where((labels2 == 'Asthma') | (labels2 == 'LRTI'))[0], axis=0)
#labels2 = np.delete(labels, np.where((labels != 'Healthy'))[0], axis=0)
#features2 = np.delete(features, np.where((labels != 'Healthy'))[0], axis=0)

unique_elements, counts_elements = np.unique(labels1, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))


le = LabelEncoder()
i_labels = le.fit_transform(labels1)
i_labels2 = le.fit_transform(img_labels1)
#print('i_labels.shape=', i_labels.shape)

oh_labels = to_categorical(i_labels)
oh_labels2 = to_categorical(i_labels2)

#print('spt_features1.shape1=', spt_features1.shape)
spt_features1 = np.reshape(spt_features1, (*spt_features1.shape, 1))
img_features1 = np.reshape(img_features1, (*img_features1.shape, 1))
#print('img_features1', img_features1.shape)

#print("features1_reshape=", features1.shape)
spt_train, spt_test, label_train, label_test = train_test_split(spt_features1, oh_labels, stratify=i_labels,
                                                    test_size=0.2, random_state = 42)
img_train, img_test, label2_train, label2_test = train_test_split(img_features1, oh_labels2, stratify=i_labels2,
                                                    test_size=0.2, random_state = 42)

spt_train = np.transpose(spt_train, (0,3,1,2))
spt_test = np.transpose(spt_test, (0,3,1,2))
img_train = np.transpose(img_train, (0,3,1,2))
img_test = np.transpose(img_test, (0,3,1,2))
df = pd.DataFrame(spt_test[10,:,:,1])
print(df)
count_t = time.time()
df.to_excel("/home/iichsk/workspace/wavpreprocess/spt_test.xlsx", sheet_name = 'sheet1')

#print('img_test=', label2_test)
print('spt_test',spt_test)
print('label_test',label_test)
print('img_test',spt_test)
print('label_test',label2_test)

#spt_train = torch.tensor(spt_train, dtype=torch.float32).to("cuda:0")
#spt_test = torch.tensor(spt_test, dtype=torch.float32).to("cuda:0")
#label_train = torch.tensor(label_train, dtype=torch.int64).to("cuda:0")
#label_test = torch.tensor(label_test, dtype=torch.int64).to("cuda:0")
#train_dataset = TensorDataset(x_train, y_train)
#test_dataset = TensorDataset(x_test, y_test)

#data loader
'''train_loader = DataLoader(train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)
val_loader = DataLoader(test_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)
test_loader = DataLoader(test_dataset,
                                         batch_size=1,
                                         shuffle=False)'''

# construct model on cuda if available
use_cuda = torch.cuda.is_available()

# loss
criterion = nn.MultiLabelSoftMarginLoss()
class spt_cnn(nn.Module):

    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(spt_cnn, self).__init__()

        self.spt_conv1 = nn.Conv2d(1, 16, kernel_size=5)  # 16@39*861
        self.spt_pool1 = nn.MaxPool2d(2)  # 16@19*430
        self.spt_dropout1 = nn.Dropout(spt_dropout)
        self.spt_conv2 = nn.Conv2d(16, 32, kernel_size=3)  # 32@18*429
        self.spt_pool2 = nn.MaxPool2d(2)  # 32@9*214
        self.spt_dropout2 = nn.Dropout(spt_dropout)
        self.spt_conv3 = nn.Conv2d(32, 64, kernel_size=2)  # 64@8*213
        self.spt_pool3 = nn.MaxPool2d(2)  # 64@4*106
        self.spt_dropout3 = nn.Dropout(spt_dropout)
        self.spt_conv4 = nn.Conv2d(64, 128, kernel_size=2)  # 128@3*105
        self.spt_pool4 = nn.MaxPool2d(2)  # 128@1*52
        self.spt_dropout4 = nn.Dropout(spt_dropout)
        self.spt_conv5 = nn.Conv2d(128, 128, kernel_size=2)  # 128@3*105
        self.spt_pool5 = nn.MaxPool2d(2)  # 128@1*52
        self.spt_dropout5 = nn.Dropout(spt_dropout)
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
        #spt_x = self.spt_relu(self.spt_conv5(spt_x))
        #spt_x = self.spt_pool5(spt_x)
        #spt_x = self.spt_dropout5(spt_x)
        spt_x = self.spt_global_pool(spt_x) # batchsize x netdim x 1 x 1
        spt_feature_x = spt_x
        spt_x = spt_x.view(spt_x.size(0), -1) # batchsize x netdim 

        spt_x = self.spt_fc1(spt_x) 
        spt_x = self.spt_bn1(spt_x)
        spt_x = self.spt_dropout4(spt_x) 

        

        spt_x = self.spt_fc2(spt_x)

        return spt_x, spt_feature_x

spt_cnn = spt_cnn()
spt_cnn.cuda()
spt_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/sym_mfcc/symptom_wav_cnn_weight_4_1000_128_1626675124.4169002.pt"))
spt_optimizer = optim.Adam(spt_cnn.parameters(), lr=learning_rate)

class cnn(nn.Module):

    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(cnn, self).__init__()

        self.spt_conv1 = nn.Conv2d(1, 16, kernel_size=4)  # 16@39*861
        self.spt_pool1 = nn.MaxPool2d(2)  # 16@19*430
        self.spt_dropout1 = nn.Dropout(img_dropout)
        self.spt_conv2 = nn.Conv2d(16, 32, kernel_size=2)  # 32@18*429
        self.spt_pool2 = nn.MaxPool2d(2)  # 32@9*214
        self.spt_dropout2 = nn.Dropout(img_dropout)
        self.spt_conv3 = nn.Conv2d(32, 64, kernel_size=2)  # 64@8*213
        self.spt_pool3 = nn.MaxPool2d(2)  # 64@4*106
        self.spt_dropout3 = nn.Dropout(img_dropout)
        self.spt_conv4 = nn.Conv2d(64, 128, kernel_size=2)  # 128@3*105
        self.spt_pool4 = nn.MaxPool2d(2)  # 128@1*52
        self.spt_dropout4 = nn.Dropout(img_dropout)
        self.spt_conv5 = nn.Conv2d(128, 256, kernel_size=2)  # 128@3*105
        self.spt_pool5 = nn.MaxPool2d(2)  # 128@1*52
        self.spt_dropout5 = nn.Dropout(img_dropout)
        
        self.spt_global_pool = nn.AdaptiveAvgPool2d(1)
        self.spt_fc1 = nn.Linear(256, 128)
        self.spt_bn1 = torch.nn.BatchNorm1d(128)
        self.spt_fcdropout1 = nn.Dropout(img_dropout)

        self.spt_fc2 = nn.Linear(128, classes) 
        
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
        spt_x = self.spt_relu(self.spt_conv5(spt_x))
        spt_x = self.spt_pool5(spt_x)  #
        spt_x = self.spt_dropout5(spt_x)
        
        spt_x = self.spt_global_pool(spt_x) # batchsize x netdim x 1 x 1
        spt_x = spt_x.view(spt_x.size(0), -1) # batchsize x netdim 

        spt_x = self.spt_fc1(spt_x) 
        spt_x = self.spt_relu(spt_x)
        spt_x = self.spt_bn1(spt_x)
        spt_feature_x = spt_x

        spt_x = self.spt_fcdropout1(spt_x)
        spt_x = self.spt_fc2(spt_x)
        #spt_x = self.spt_relu(spt_x)

        #spt_x = self.spt_bn2(spt_x)
        #spt_x = self.spt_fc3(spt_x)
        return spt_x, spt_feature_x

# hyper-parameters
img_cnn = cnn()
img_cnn.cuda()
img_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/sym_scalo/symptom_scalo_cnn_weight_4_300_128_1626674999.2049477.pt"))

img_optimizer = optim.Adam(img_cnn.parameters(), lr=learning_rate)


# hyper-parameters
class concat_fc(nn.Module):
    def __init__(self):
        super(concat_fc, self).__init__()

       
        self.concat_fc1 = nn.Linear(256, 128)
        self.concat_fc2 = nn.Linear(128, 64)
        self.concat_fc3 = nn.Linear(64, classes)


        self.concat_relu = nn.ReLU()
        self.concat_bn1 = torch.nn.BatchNorm1d(128)
        self.concat_bn2 = torch.nn.BatchNorm1d(64)


        self.concat_dropout1 = nn.Dropout(0.35)

        self.concat_dropout2 = nn.Dropout(0.35)

       
    
    def forward(self, concat_x):
    
        
        #concat_x = concat_x.view(concat_x.size(0), -1)
        concat_x = self.concat_fc1(concat_x)  #
        concat_x = self.concat_relu(concat_x)
        concat_x = self.concat_bn1(concat_x)
        concat_x = self.concat_dropout1(concat_x)
        concat_x = self.concat_fc2(concat_x)  #
        concat_x = self.concat_relu(concat_x)
        concat_x = self.concat_bn2(concat_x)
        concat_x = self.concat_dropout2(concat_x)
        concat_x = self.concat_fc3(concat_x)  #

        return concat_x


concat_fc = concat_fc()
concat_fc.cuda()
#print('model-------', cnn)
# backpropagation method
concat_optimizer = optim.Adam(concat_fc.parameters(), lr=learning_rate)
z = np.random.permutation(spt_train.shape[0])

def spt_cnn_train():
    num_epochs=2000

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
            spt_optimizer.zero_grad()
            model_output, _ = spt_cnn(spt_input)
            # calculate loss
            loss = criterion(model_output, label)
            # back propagation
            loss.backward()
            # weight update
            spt_optimizer.step()

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
    torch.save(spt_cnn.state_dict(), "/home/iichsk/workspace/weight/symptom_spt_cnn_weight_{}_{}_{}_{}.pt".format(classes, num_epochs, batch_size, count_t))


def concat_train(concat_trainset,feature_label):
    num_epochs = 350
    z = np.random.permutation(concat_trainset.shape[0])

    trn_loss_list = []
    print('concat train start!!!!!')
    concat_fc.train()
    for epoch in range(num_epochs):
        trn_loss = 0.0
        for i in range(int(concat_trainset.shape[0] / batch_size)):
            #spt_input = torch.Tensor(spt_train[z[i*batch_size:(i+1)* batch_size], :, :, :]).cuda()
            concat_input = torch.Tensor(concat_trainset[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            concat_input = Variable(concat_input.data, requires_grad=True)

            label =  torch.Tensor(feature_label[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            # grad init
            concat_optimizer.zero_grad()
            model_output = concat_fc(concat_input)
            
            # calculate loss

            concat_loss = criterion(model_output, label)
            # back propagation
            concat_loss.backward(retain_graph=True)

            # weight update
            concat_optimizer.step()

            # trn_loss summary
            trn_loss += concat_loss.item()
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
    torch.save(concat_fc.state_dict(), "/home/iichsk/workspace/weight/concat_fc_weight_{}_{}_{}.pt".format(num_epochs, batch_size, count_t))

def spt_cnn_test(spt_data):
    spt_cnn.eval()
    test_loss = 0.0
    predictions = np.zeros((int(spt_data.shape[0]), classes))
    spt_feature = np.zeros((int(spt_data.shape[0]), 128,1,1))
    correct = 0
    wrong = 0
    for j in range(int(spt_data.shape[0])):
        spt_input = torch.Tensor(spt_data[j:(j+1), :, :, :]).cuda()
        test_output, spt_feature_x = spt_cnn(spt_input)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        spt_feature[j:j+1,:] = spt_feature_x.data.cpu().numpy()

    return predictions, spt_feature

def img_cnn_test(img_data):
    img_cnn.eval()
    test_loss = 0.0
    predictions = np.zeros((int(img_data.shape[0]), classes))
    img_feature = np.zeros((int(img_data.shape[0]), 128))

    for j in range(int(img_data.shape[0])):
        img_input = torch.Tensor(img_data[j:(j+1), :, :, :]).cuda()
        test_output, img_feature_x = img_cnn(img_input)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        img_feature[j:j+1,:] = img_feature_x.data.cpu().numpy()

    return predictions, img_feature

def concat_test(concat_testset):
    concat_fc.eval()
    test_loss = 0.0
    predictions = np.zeros((int(concat_testset.shape[0]), classes))
   
    for j in range(int(concat_testset.shape[0])):
        concat_input = torch.Tensor(concat_testset[j:(j+1), :]).cuda()
        test_output = concat_fc(concat_input)

        predictions[j:j+1,:] = test_output.detach().cpu().numpy()

    return predictions


#spectrum part
#spt_cnn_train()
_, spt_feature_x = spt_cnn_test(spt_train)
spt_prediction, spt_feature = spt_cnn_test(spt_test)
classpreds1 = np.argmax(spt_prediction, axis=1)  # predicted classes
target = np.argmax(label_test, axis=1)  # true classes
c_names = ['normal','crackles','wheezes','Both']# Classification Report
print(classification_report(target, classpreds1, target_names=c_names))
# Confusion Matrix
print(confusion_matrix(target, classpreds1))
print('MFCC classprediction= ')
print(classpreds1)

#waveform part
_, img_feature_x = img_cnn_test(img_train)
img_prediction, img_feature = img_cnn_test(img_test)
classpreds2 = np.argmax(img_prediction, axis=1)  # predicted classes
target = np.argmax(label_test, axis=1)  # true classes
c_names = ['normal','crackles','wheezes','Both']# Classification Report
print(classification_report(target, classpreds2, target_names=c_names))
# Confusion Matrix
print(confusion_matrix(target, classpreds2))
print('Scalo classprediction= ')
print(classpreds2)

#concat data part
spt_feature_x = spt_feature_x.reshape(spt_feature_x.shape[0],spt_feature.shape[1])
concat_trainset = np.concatenate((spt_feature_x, img_feature_x),1)
concat_train(concat_trainset,label_train)

spt_feature = spt_feature.reshape(spt_feature.shape[0],spt_feature.shape[1])
concat_testset = np.concatenate((spt_feature, img_feature),1)
predictions = concat_test(concat_testset)

classpreds = np.argmax(predictions, axis=1)  # predicted classes
print('classpreds', classpreds)
target = np.argmax(label_test, axis=1)  # true classes
print('target', target)
#c_names = ['Abnormal', 'Healthy']
c_names = ['normal','crackles','wheezes','Both']
# Classification Report
print(classification_report(target, classpreds, target_names=c_names))
# Confusion Matrix
print(confusion_matrix(target, classpreds))

classpreds1=np.expand_dims(classpreds1, axis=1)
classpreds2=np.expand_dims(classpreds2, axis=1)
classpreds=np.expand_dims(classpreds, axis=1)
target=np.expand_dims(target, axis=1)

concat_pred = np.concatenate((classpreds,target),1)
concat_pred = np.concatenate((classpreds1, classpreds2, classpreds,target),1)
df = pd.DataFrame(concat_pred,columns=['MFCC','Waveform','concat','target'])
print(df)
count_t = time.time()
df.to_excel("/home/iichsk/workspace/wavpreprocess/predict_value_{}.xlsx".format(count_t), sheet_name = 'sheet1')
#normal error test
'''spt_prediction2, spt_feature2 = spt_cnn_test(spt_test2)
img_prediction2, img_feature2 = img_cnn_test(img_test2)
concat_testset2 = np.concatenate((spt_feature2, img_feature2),1)

predictions2 = concat_test(concat_testset2)
classpreds2 = np.argmax(predictions2, axis=1)  # predicted classes
print('prediction=', classpreds2)

target2 = np.argmax(label_test2, axis=1)  # true classes
print('target=', target2)

#c_names = ['Abnormal', 'Healthy']
c_names2 = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Pneumonia', 'URTI', 'Healthy']

# Classification Report
print(classification_report(target2, classpreds2, target_names=c_names2))
# Confusion Matrix
print(confusion_matrix(target2, classpreds2))
'''