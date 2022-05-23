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
filter_size3 = (2,10)
filter_size4 = (2, 10)
classes = 4
img_dropout = 0.3

mypath = "/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/processed_audio_files_nopad"
mypath_waveform = "/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/waveform_image"

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]

p_id_in_file = [] # patient IDs corresponding to each file

for name in filenames:
    p_id_in_file.append(int(name.split('_')[0]))
p_id_in_file = np.array(p_id_in_file)

def extract_image(file_name):
    im = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    return im

filepaths_waveform = [join(mypath_waveform, f)+'_waveform.png' for f in filenames] # full paths of files
p_diag = pd.read_csv("/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/csv_data/data2.csv",header=None)

labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])

img_features = []
for file_name2 in filepaths_waveform:
    data = extract_image(file_name2)
    #print("file_name=",file_name2)
    img_features.append(data)

img_features = np.array(img_features)
img_features1 = np.delete(img_features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)

unique_elements, counts_elements = np.unique(labels, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

le = LabelEncoder()
i_labels = le.fit_transform(labels)

oh_labels = to_categorical(i_labels)
img_features1 = np.reshape(img_features1, (*img_features1.shape, 1))

waveform_train, waveform_test, label_train, label_test = train_test_split(img_features1, oh_labels, stratify=i_labels,
                                                    test_size=0.2, random_state = 42)
waveform_train = np.transpose(waveform_train, (0,3,1,2))
waveform_test = np.transpose(waveform_test, (0,3,1,2))
print('label_test',label_test[:5])
use_cuda = torch.cuda.is_available()

# loss
criterion = nn.MultiLabelSoftMarginLoss()

class img_cnn(nn.Module):

    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(img_cnn, self).__init__()

        self.img_conv1 = nn.Conv2d(1, 16, kernel_size=filter_size3)  # 16@39*861
        self.img_pool1 = nn.MaxPool2d(2)  # 16@19*430
        self.img_dropout1 = nn.Dropout(img_dropout)

        self.img_conv2 = nn.Conv2d(16, 32, kernel_size=filter_size4)  # 32@18*429
        self.img_pool2 = nn.MaxPool2d(2)  # 32@9*214
        self.img_dropout2 = nn.Dropout(img_dropout)

        self.img_conv3 = nn.Conv2d(32, 64, kernel_size=filter_size4)  # 64@8*213
        self.img_pool3 = nn.MaxPool2d(2)  # 64@4*106
        self.img_dropout3 = nn.Dropout(img_dropout)

        self.img_conv4 = nn.Conv2d(64, 128, kernel_size=2)  # 64@8*213
        self.img_pool4 = nn.MaxPool2d(2)  # 64@4*106
        self.img_dropout4 = nn.Dropout(img_dropout)

        self.img_conv5 = nn.Conv2d(128, 256, kernel_size=2)  # 64@8*213
        self.img_pool5 = nn.MaxPool2d(2)  # 64@4*106
        self.img_dropout5 = nn.Dropout(img_dropout)

        self.img_conv6 = nn.Conv2d(256, 512, kernel_size=2)  # 64@8*213
        self.img_pool6 = nn.MaxPool2d(2)  # 64@4*106
        self.img_dropout6 = nn.Dropout(img_dropout)

        self.img_conv7 = nn.Conv2d(512, 1024, kernel_size=2)  # 64@8*213
        self.img_pool7 = nn.MaxPool2d(2)  # 64@4*106
        self.img_dropout7 = nn.Dropout(img_dropout)

        self.img_global_pool = nn.AdaptiveAvgPool2d(1)

        self.img_fc1 = nn.Linear(1024, 512)
        self.img_bn1 = torch.nn.BatchNorm1d(512)

        self.img_fc2 = nn.Linear(512, 128)
        self.img_fc3 = nn.Linear(128, classes)

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
        img_x = self.img_dropout5(img_x)  #

        img_x = self.img_relu(self.img_conv6(img_x))
        img_x = self.img_pool6(img_x)  #
        img_x = self.img_dropout6(img_x)  #

        img_x = self.img_relu(self.img_conv7(img_x))
        #img_x = self.img_pool7(img_x)  #
        img_x = self.img_dropout7(img_x)  #

        img_x = self.img_global_pool(img_x)

        img_x = img_x.view(img_x.size(0), -1)
        

        img_x = self.img_fc1(img_x)  #
        #img_x = self.img_bn1(img_x)
        
        img_x = self.img_fc2(img_x)
        img_feature_x=img_x
        img_x = self.img_fc3(img_x)

        return img_x, img_feature_x


img_cnn = img_cnn()
img_cnn.cuda()
# img_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/sym_wave/symptom_wav_cnn_weight_4_1000_128_1626951295.9009268.pt"))
img_optimizer = optim.Adam(img_cnn.parameters(), lr=learning_rate)

def img_cnn_train(wave_trainset,feature_label):
    num_epochs = 300
    z = np.random.permutation(wave_trainset.shape[0])
    trn_loss_list = []
    #wave_trainset=np.reshape(wave_trainset, (*wave_trainset.shape, 1))
    print('concat train start!!!!!')
    img_cnn.train()
    for epoch in range(num_epochs):
        trn_loss = 0.0
        for i in range(int(wave_trainset.shape[0] / batch_size)):
            wave_input = torch.Tensor(wave_trainset[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            wave_input = Variable(wave_input.data, requires_grad=True)
            label =  torch.Tensor(feature_label[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            
            # grad init
            img_optimizer.zero_grad()
            model_output,_ = img_cnn(wave_input)
            # calculate loss
            print('model_output',model_output.shape)
            print('labe',label.shape)
            
            wave_loss = criterion(model_output, label)
            # back propagation
            wave_loss.backward(retain_graph=True)
            # weight update
            img_optimizer.step()
            # trn_loss summary
            trn_loss += wave_loss.item()
           # del (memory issue)
            '''del loss
            del model_output'''

            # 학습과정 출력
        if (epoch + 1) % 50 == 0:  #
            print("epoch: {}/{} | trn loss: {:.8f}".format(
                epoch + 1, num_epochs, trn_loss / 100
            ))
            trn_loss_list.append(trn_loss / 100)
    count_t = time.time()
    torch.save(img_cnn.state_dict(), "/home/iichsk/workspace/weight/img_cnn_weight_{}_{}_{}.pt".format(num_epochs, batch_size, count_t))

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

###################waveform part###################
np.save('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/waveform_train',waveform_train)
waveform_train = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/waveform_train.npy')
np.save('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/waveform_test',waveform_test)
waveform_test = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/waveform_test.npy')
label_train = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/label_train.npy')
label_test = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/label_test.npy')
print('label_test',label_test[:5])
img_cnn_train(waveform_train,label_train)
img_prediction, _ = img_cnn_test(waveform_test)


classpreds = np.argmax(img_prediction, axis=1)  # predicted classes
target = np.argmax(label_test, axis=1)  # true classes
c_names = ['normal','crackles','wheezes','Both']# Classification Report\
print('#'*10,'Waveform CNN report','#'*10)
print(classification_report(target, classpreds3, target_names=c_names))
# Confusion Matrix
print(confusion_matrix(target, classpreds3))

################################################


target = np.argmax(label_test, axis=1)  # true classes
print('target', target)
#c_names = ['Abnormal', 'Healthy']
c_names = ['normal','wheezes','crackles','Both']
# Classification Report
print('#'*10,'wave CNN report','#'*10)
print(classification_report(target, classpreds, target_names=c_names))
# Confusion Matrix
print(confusion_matrix(target, classpreds))

classpreds=np.expand_dims(classpreds, axis=1)
target=np.expand_dims(target, axis=1)

wave_pred = np.concatenate((classpreds,target),1)

df = pd.DataFrame(wave_pred,columns=['wave','target'])
count_t = time.time()
df.to_excel("/home/iichsk/workspace/wavpreprocess/triple_wave_predict_value_{}.xlsx".format(count_t), sheet_name = 'sheet1')
