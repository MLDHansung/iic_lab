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
from sklearn.preprocessing import MaxAbsScaler


batch_size = 128
learning_rate = 1e-3
filter_size1 = 5
filter_size2 = 3
filter_size3 = (2,10)
filter_size4 = (2, 10)
classes = 4
spt_dropout = 0.3
scalo_dropout = 0.3
img_dropout = 0.3

mypath = "/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/processed_audio_files_nopad"
mypath_waveform = "/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/waveform_image"

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]
filenames2 = [f for f in listdir(mypath_waveform) if (isfile(join(mypath_waveform, f)) and f.endswith('.png'))]

p_id_in_file_waveform = [] # patient IDs corresponding to each file


for name in filenames:
    p_id_in_file_waveform.append(int(name.split('_')[0]))
p_id_in_file_waveform = np.array(p_id_in_file_waveform)
 
def extract_image(file_name):
    im = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

    return im

filepaths_waveform = [join(mypath_waveform, f)+'_waveform.png' for f in filenames] # full paths of files
p_diag = pd.read_csv("/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/csv_data/data2.csv",header=None)

labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file_waveform])



img_features = []

for file_name2 in filepaths_waveform:
    data = extract_image(file_name2)
    img_features.append(data)

print('Finished feature extraction from ', len(img_features), ' files')
labels1 = np.delete(labels, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)
img_features = np.array(img_features)
img_features1 = np.delete(img_features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)
le = LabelEncoder()
i_labels = le.fit_transform(labels1)
oh_labels = to_categorical(i_labels)
img_features1 = np.reshape(img_features1, (*img_features1.shape, 1))
waveform_train, waveform_test, label3_train, label3_test = train_test_split(img_features1, oh_labels, stratify=i_labels,
                                                    test_size=0.2, random_state = 42)

waveform_test = np.transpose(waveform_test, (0,3,1,2))

use_cuda = torch.cuda.is_available()

# loss
criterion = nn.MultiLabelSoftMarginLoss()


class concat_fc(nn.Module):
    def __init__(self):
        super(concat_fc, self).__init__()

        self.concat_fc1 = nn.Linear(384, 128)
        self.concat_fc2 = nn.Linear(128, 64)
        self.concat_fc3 = nn.Linear(64, classes)
        self.concat_relu = nn.ReLU()
        self.concat_spt_bn1 = torch.nn.BatchNorm1d(128)
        self.concat_scalo_bn1 = torch.nn.BatchNorm1d(128)
        self.concat_wave_bn1 = torch.nn.BatchNorm1d(128)
        self.concat_bn1 = torch.nn.BatchNorm1d(128)

        self.concat_bn2 = torch.nn.BatchNorm1d(64)
        self.concat_dropout1 = nn.Dropout(0.30)
        self.concat_dropout2 = nn.Dropout(0.30)

    def forward(self, spt_input, scalo_input, wave_input):   
        spt_raw = spt_input
        scalo_raw = scalo_input
        wave_raw = wave_input

        spt_input = self.concat_spt_bn1(spt_input)
        spt_bn = spt_input
        scalo_input = self.concat_scalo_bn1(scalo_input)
        scalo_bn = scalo_input
        wave_input = self.concat_wave_bn1(wave_input)
        wave_bn = wave_input

        concat_x = torch.cat((spt_input, scalo_input, wave_input),1)
        
        concat_x = self.concat_fc1(concat_x)  #
        concat_x = self.concat_bn1(concat_x)
        concat_feature = concat_x
        concat_x = self.concat_fc2(concat_x)  #
        concat_x = self.concat_bn2(concat_x)
        concat_x = self.concat_fc3(concat_x)  #

        return concat_x, concat_feature, spt_bn, scalo_bn, wave_bn, spt_raw, scalo_raw, wave_raw



concat_fc = concat_fc()
concat_fc.cuda()
concat_fc.load_state_dict(torch.load("/home/iichsk/workspace/weight/concat_fc_weight_300_128_1631350767.6301105.pt"))

concat_optimizer = optim.Adam(concat_fc.parameters(), lr=learning_rate)

def concat_train(spt_feature,scalo_feature,wave_feature,feature_label):
    num_epochs =  300
    z = np.random.permutation(spt_feature.shape[0])
    trn_loss_list = []
    print('concat train start!!!!!')
    concat_fc.train()
    for epoch in range(num_epochs):
        trn_loss = 0.0
        for i in range(int(spt_feature.shape[0] / batch_size)):
            spt_input = torch.Tensor(spt_feature[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            scalo_input = torch.Tensor(scalo_feature[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            wave_input = torch.Tensor(wave_feature[z[i*batch_size:(i+1)* batch_size], :]).cuda()


            # concat_input = Variable(concat_input.data, requires_grad=True)
            label =  torch.Tensor(feature_label[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            
            # grad init
            concat_optimizer.zero_grad()
            model_output,_ , _, _, _,_,_,_= concat_fc(spt_input,scalo_input,wave_input)
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
        if (epoch + 1) % 50 == 0:  #
            print("epoch: {}/{} | trn loss: {:.8f}".format(
                epoch + 1, num_epochs, trn_loss / 100
            ))
            trn_loss_list.append(trn_loss / 100)
    count_t = time.time()
    torch.save(concat_fc.state_dict(), "/home/iichsk/workspace/weight/concat_fc_weight_{}_{}_{}.pt".format(num_epochs, batch_size, count_t))

def concat_test(spt_feature, scalo_feature, wave_feature):
    concat_fc.eval()
    test_loss = 0.0
    predictions = np.zeros((int(spt_feature.shape[0]), classes))
    concat_features = np.zeros((int(spt_feature.shape[0]), 128))
    mfcc_features = np.zeros((int(spt_feature.shape[0]), 128))
    scalogram_features = np.zeros((int(scalo_feature.shape[0]), 128))
    waveform_features = np.zeros((int(wave_feature.shape[0]), 128))

    mfcc_features_raw = np.zeros((int(spt_feature.shape[0]), 128))
    scalogram_features_raw  = np.zeros((int(scalo_feature.shape[0]), 128))
    waveform_features_raw  = np.zeros((int(wave_feature.shape[0]), 128))

    for j in range(int(spt_feature.shape[0])):
        spt_input = torch.Tensor(spt_feature[j:(j+1), :]).cuda()
        scalo_input = torch.Tensor(scalo_feature[j:(j+1), :]).cuda()
        wave_input = torch.Tensor(wave_feature[j:(j+1), :]).cuda()

        test_output, concat_feature, mfcc_feature, scalogram_feature, waveform_feature, mfcc_raw, scalogram_raw, waveform_raw= concat_fc(spt_input,scalo_input,wave_input)        
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        concat_features[j:j+1,:] = concat_feature.detach().cpu().numpy()

        mfcc_features[j:j+1,:] = mfcc_feature.detach().cpu().numpy()
        scalogram_features[j:j+1,:] = scalogram_feature.detach().cpu().numpy()
        waveform_features[j:j+1,:] = waveform_feature.detach().cpu().numpy()

        mfcc_features_raw[j:j+1,:] = mfcc_raw.detach().cpu().numpy()
        scalogram_features_raw[j:j+1,:] = scalogram_raw.detach().cpu().numpy()
        waveform_features_raw[j:j+1,:] = waveform_raw.detach().cpu().numpy()

    return predictions,concat_features,mfcc_features,scalogram_features,waveform_features,mfcc_features_raw,scalogram_features_raw,waveform_features_raw


label_test = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/label_test.npy')

###############spectrum part###################

spt_feature_x = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/spt_feature_x_norm.npy') # MFCC train feature load
spt_feature = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/spt_feature_norm.npy') # MFCC test feature load
spt_feature = spt_feature.reshape(spt_feature.shape[0],spt_feature.shape[1])
spt_feature_x = spt_feature_x.reshape(spt_feature_x.shape[0],spt_feature_x.shape[1])

################################################

###################scalo part###################

scalo_feature_x = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/scalo_feature_x.npy') # scalogram train feature load
scalo_feature = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/scalo_feature.npy') # scalogram test feature load

################################################

###################waveform part###################

# wave_prediction, wave_feature = img_cnn_test(waveform_test)
# np.save('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/wave_feature_test5',wave_feature)
wave_feature_x = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/img_feature_x2.npy') # waveform train feature load
wave_feature = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/img_feature_test5.npy') # waveform test feature load

################################################

####################concat data part###################

label_train = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/label_train.npy')
label_test = np.load('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/label_test.npy')

##############train################################

# concat_train(spt_feature_x, scalo_feature_x, wave_feature_x, label_train)

##############test################################

predictions, concat_feature, mfcc_features, scalogram_features, waveform_features, mfcc_raw, scalo_raw, waveform_raw = concat_test(spt_feature, scalo_feature, wave_feature)
concat_feature = np.reshape(concat_feature, (*concat_feature.shape, 1))

df2 = pd.DataFrame(mfcc_features)
count_t = time.time()
df2.to_excel("/home/iichsk/workspace/wavpreprocess/mfcc_feature_fc_bn_each{}.xlsx".format(count_t), sheet_name = 'sheet1')

df3 = pd.DataFrame(scalogram_features)
count_t = time.time()
df3.to_excel("/home/iichsk/workspace/wavpreprocess/scalo_feature_fc_bn_each{}.xlsx".format(count_t), sheet_name = 'sheet1')

df4 = pd.DataFrame(waveform_features)
count_t = time.time()
df4.to_excel("/home/iichsk/workspace/wavpreprocess/waveform_feature_fc_bn_each{}.xlsx".format(count_t), sheet_name = 'sheet1')

df5 = pd.DataFrame(mfcc_raw)
count_t = time.time()
df5.to_excel("/home/iichsk/workspace/wavpreprocess/mfcc_feature_raw{}.xlsx".format(count_t), sheet_name = 'sheet1')

df6 = pd.DataFrame(scalo_raw)
count_t = time.time()
df6.to_excel("/home/iichsk/workspace/wavpreprocess/scalo_feature_raw{}.xlsx".format(count_t), sheet_name = 'sheet1')

df7 = pd.DataFrame(waveform_raw)
count_t = time.time()
df7.to_excel("/home/iichsk/workspace/wavpreprocess/waveform_feature_raw{}.xlsx".format(count_t), sheet_name = 'sheet1')




classpreds = np.argmax(predictions, axis=1)  # predicted classes
target = np.argmax(label_test, axis=1)  # true classes
print('target', target)
c_names = ['normal','wheezes','crackles','Both']

# Classification Report
print('#'*10,'Triple Ensemble CNN1 report','#'*10)
print(classification_report(target, classpreds, target_names=c_names))
# Confusion Matrix
print(confusion_matrix(target, classpreds))
