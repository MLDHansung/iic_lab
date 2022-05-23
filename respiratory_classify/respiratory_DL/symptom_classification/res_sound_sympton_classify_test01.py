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
print('max',max(spt_features_1))
print('min',min(spt_features_1))


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
df3 = pd.DataFrame(spt_test[1,:,:,0])
count_t = time.time()
df3.to_excel("/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/mfcc_testset_data_{}.xlsx".format(count_t), sheet_name = 'sheet1')

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

        self.spt_conv1 = nn.Conv2d(1, 16, kernel_size=filter_size1)  # 16@39*861
        self.spt_pool1 = nn.MaxPool2d(2)  # 16@19*430
        self.spt_dropout1 = nn.Dropout(dropout)
        self.spt_conv2 = nn.Conv2d(16, 32, kernel_size=filter_size2)  # 32@18*429
        self.spt_pool2 = nn.MaxPool2d(2)  # 32@9*214
        self.spt_dropout2 = nn.Dropout(dropout)
        self.spt_conv3 = nn.Conv2d(32, 64, kernel_size=2)  # 64@8*213
        self.spt_pool3 = nn.MaxPool2d(2)  # 64@4*106
        self.spt_dropout3 = nn.Dropout(dropout)
        self.spt_conv4 = nn.Conv2d(64, 128, kernel_size=2)  # 128@3*105
        self.spt_pool4 = nn.MaxPool2d(2)  # 128@1*52
        self.spt_dropout4 = nn.Dropout(dropout)
        self.spt_conv5 = nn.Conv2d(128, 128, kernel_size=2)  # 128@3*105
        self.spt_pool5 = nn.MaxPool2d(2)  # 128@1*52
        self.spt_dropout5 = nn.Dropout(dropout)
        self.spt_global_pool = nn.AdaptiveAvgPool2d(1)
        self.spt_fc1 = nn.Linear(128, 64)
        self.spt_bn1 = torch.nn.BatchNorm1d(64)
        self.spt_fc2 = nn.Linear(64, classes)
        self.spt_relu = nn.ReLU()
    
    def forward(self, spt_x):

        feature1 = spt_x

        spt_x = self.spt_relu(self.spt_conv1(spt_x))

        feature2 = spt_x
        spt_x = self.spt_pool1(spt_x)  #
        feature3 = spt_x
        spt_x = self.spt_dropout1(spt_x)
        feature4 = spt_x
        spt_x = self.spt_relu(self.spt_conv2(spt_x))
        feature5 = spt_x
        spt_x = self.spt_pool2(spt_x) 
        feature6 = spt_x
        spt_x = self.spt_dropout2(spt_x)
        feature7 = spt_x
        spt_x = self.spt_relu(self.spt_conv3(spt_x))
        feature8 = spt_x
        spt_x = self.spt_pool3(spt_x) 
        feature9 = spt_x
        spt_x = self.spt_dropout3(spt_x)  #
        feature10 = spt_x
        spt_x = self.spt_relu(self.spt_conv4(spt_x))
        feature11 = spt_x
        spt_x = self.spt_pool4(spt_x)  #
        feature12 = spt_x
        spt_x = self.spt_dropout4(spt_x)  #
        feature13 = spt_x
        spt_x = self.spt_global_pool(spt_x) # batchsize x netdim x 1 x 1
        spt_feature_x = spt_x
        spt_x = spt_x.view(spt_x.size(0), -1) # batchsize x netdim 

        spt_x = self.spt_fc1(spt_x) 
        spt_x = self.spt_bn1(spt_x)
        spt_x = self.spt_dropout4(spt_x) 

        

        spt_x = self.spt_fc2(spt_x)

        return spt_x, spt_feature_x, feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12, feature13


spt_cnn = spt_cnn()
spt_cnn.cuda()
spt_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/symptom_wav_cnn_weight_4_2000_128_1630986830.5446396.pt"))

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
            model_output, feature_x, _, _, _, _, _, _, _, _, _, _, _, _, _ = spt_cnn(spt_input)
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
    mfcc_features1 = np.zeros((int(spt_test.shape[0]), 40, 295))
    mfcc_features2 = np.zeros((int(spt_test.shape[0]), 36, 291))
    mfcc_features3 = np.zeros((int(spt_test.shape[0]), 18, 145))
    mfcc_features4 = np.zeros((int(spt_test.shape[0]), 18, 145))
    mfcc_features5 = np.zeros((int(spt_test.shape[0]), 16, 143))
    mfcc_features6 = np.zeros((int(spt_test.shape[0]), 8, 71))
    mfcc_features7 = np.zeros((int(spt_test.shape[0]), 8, 71))
    mfcc_features8 = np.zeros((int(spt_test.shape[0]), 7, 70))
    mfcc_features9 = np.zeros((int(spt_test.shape[0]), 3, 35))
    mfcc_features10 = np.zeros((int(spt_test.shape[0]), 3, 35))
    mfcc_features11 = np.zeros((int(spt_test.shape[0]), 2, 34))
    mfcc_features12 = np.zeros((int(spt_test.shape[0]), 1, 17))
    mfcc_features13 = np.zeros((int(spt_test.shape[0]), 1, 17))

    correct = 0
    wrong = 0
    for j in range(int(spt_test.shape[0])):
        #spt_input = torch.Tensor(spt_test[j:(j+1), :, :, :]).cuda()
        spt_input = torch.Tensor(spt_test[j:(j+1), :, :, :]).cuda()

        test_label =  torch.Tensor(label_test[j:(j+1), :]).cuda()
        test_output, feature, feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12, feature13 = spt_cnn(spt_input)
        feature = np.squeeze(feature)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        mfcc_features[j:j+1,:] = feature.detach().cpu().numpy()
        mfcc_features1[j:j+1,:,:] = feature1[:,0,:,:].detach().cpu().numpy()
        mfcc_features2[j:j+1,:,:] = feature2[:,7,:,:].detach().cpu().numpy()
        mfcc_features3[j:j+1,:,:] = feature3[:,7,:,:].detach().cpu().numpy()
        mfcc_features4[j:j+1,:,:] = feature4[:,7,:,:].detach().cpu().numpy()
        mfcc_features5[j:j+1,:,:] = feature5[:,15,:,:].detach().cpu().numpy()
        mfcc_features6[j:j+1,:,:] = feature6[:,15,:,:].detach().cpu().numpy()
        mfcc_features7[j:j+1,:,:] = feature7[:,15,:,:].detach().cpu().numpy()
        mfcc_features8[j:j+1,:,:] = feature8[:,63,:,:].detach().cpu().numpy()
        mfcc_features9[j:j+1,:,:] = feature9[:,63,:,:].detach().cpu().numpy()
        mfcc_features10[j:j+1,:,:] = feature10[:,63,:,:].detach().cpu().numpy()
        mfcc_features11[j:j+1,:,:] = feature11[:,127,:,:].detach().cpu().numpy()
        mfcc_features12[j:j+1,:,:] = feature12[:,127,:,:].detach().cpu().numpy()
        mfcc_features13[j:j+1,:,:] = feature13[:,127,:,:].detach().cpu().numpy()
        
        
    return predictions, mfcc_features, mfcc_features1, mfcc_features2, mfcc_features3, mfcc_features4, mfcc_features5, mfcc_features6, mfcc_features7, mfcc_features8, mfcc_features9, mfcc_features10, mfcc_features11, mfcc_features12, mfcc_features13


# train()
print('train finished!!!')
_, mfcc_features_x, _, _, _, _, _, _, _, _, _, _, _, _, _ = test(spt_train)
np.save('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/spt_feature_x_norm',mfcc_features_x)

predictions, mfcc_features, mfcc_features1, mfcc_features2, mfcc_features3, mfcc_features4, mfcc_features5, mfcc_features6, mfcc_features7, mfcc_features8, mfcc_features9, mfcc_features10, mfcc_features11, mfcc_features12, mfcc_features13 = test(spt_test)
np.save('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/feature_data/spt_feature_norm',mfcc_features)

mfcc_features = np.squeeze(mfcc_features)
df = pd.DataFrame(mfcc_features)
count_t = time.time()
df.to_excel("/home/iichsk/workspace/wavpreprocess/feature_data/mfcc_feature_{}.xlsx".format(count_t), sheet_name = 'sheet1')

mfcc_features1 = np.squeeze(mfcc_features1[1,:,:])
df1 = pd.DataFrame(mfcc_features1)
count_t = time.time()
df1.to_excel("/home/iichsk/workspace/wavpreprocess/feature_data/mfcc_feature1_{}.xlsx".format(count_t), sheet_name = 'sheet1')

mfcc_features2 = np.squeeze(mfcc_features2[1,:,:])
df2 = pd.DataFrame(mfcc_features2)
count_t = time.time()
df2.to_excel("/home/iichsk/workspace/wavpreprocess/feature_data/mfcc_feature2_{}.xlsx".format(count_t), sheet_name = 'sheet1')

mfcc_features3 = np.squeeze(mfcc_features3[1,:,:])
df3 = pd.DataFrame(mfcc_features3)
count_t = time.time()
df3.to_excel("/home/iichsk/workspace/wavpreprocess/feature_data/mfcc_feature3_{}.xlsx".format(count_t), sheet_name = 'sheet1')

mfcc_features4 = np.squeeze(mfcc_features4[1,:,:])
df4 = pd.DataFrame(mfcc_features4)
count_t = time.time()
df4.to_excel("/home/iichsk/workspace/wavpreprocess/feature_data/mfcc_feature4_{}.xlsx".format(count_t), sheet_name = 'sheet1')

mfcc_features5 = np.squeeze(mfcc_features5[1,:,:])
df5 = pd.DataFrame(mfcc_features5)
count_t = time.time()
df5.to_excel("/home/iichsk/workspace/wavpreprocess/feature_data/mfcc_feature5_{}.xlsx".format(count_t), sheet_name = 'sheet1')

mfcc_features6 = np.squeeze(mfcc_features6[1,:,:])
df6 = pd.DataFrame(mfcc_features6)
count_t = time.time()
df6.to_excel("/home/iichsk/workspace/wavpreprocess/feature_data/mfcc_feature6_{}.xlsx".format(count_t), sheet_name = 'sheet1')

mfcc_features7 = np.squeeze(mfcc_features7[1,:,:])
df7 = pd.DataFrame(mfcc_features7)
count_t = time.time()
df7.to_excel("/home/iichsk/workspace/wavpreprocess/feature_data/mfcc_feature7_{}.xlsx".format(count_t), sheet_name = 'sheet1')

mfcc_features8 = np.squeeze(mfcc_features8[1,:,:])
df8 = pd.DataFrame(mfcc_features8)
count_t = time.time()
df8.to_excel("/home/iichsk/workspace/wavpreprocess/feature_data/mfcc_feature8_{}.xlsx".format(count_t), sheet_name = 'sheet1')

mfcc_features9 = np.squeeze(mfcc_features9[1,:,:])
df9 = pd.DataFrame(mfcc_features9)
count_t = time.time()
df9.to_excel("/home/iichsk/workspace/wavpreprocess/feature_data/mfcc_feature9_{}.xlsx".format(count_t), sheet_name = 'sheet1')

mfcc_features10 = np.squeeze(mfcc_features10[1,:,:])
df10 = pd.DataFrame(mfcc_features10)
count_t = time.time()
df10.to_excel("/home/iichsk/workspace/wavpreprocess/feature_data/mfcc_feature10_{}.xlsx".format(count_t), sheet_name = 'sheet1')

mfcc_features11 = np.squeeze(mfcc_features11[1,:,:])
df11 = pd.DataFrame(mfcc_features11)
count_t = time.time()
df11.to_excel("/home/iichsk/workspace/wavpreprocess/feature_data/mfcc_feature11_{}.xlsx".format(count_t), sheet_name = 'sheet1')

mfcc_features12 = np.squeeze(mfcc_features12[1,:,:])
df12 = pd.DataFrame(mfcc_features12)
count_t = time.time()
df12.to_excel("/home/iichsk/workspace/wavpreprocess/feature_data/mfcc_feature12_{}.xlsx".format(count_t), sheet_name = 'sheet1')

mfcc_features13 = np.squeeze(mfcc_features13[1,:,:])
df13 = pd.DataFrame(mfcc_features13)
count_t = time.time()
df13.to_excel("/home/iichsk/workspace/wavpreprocess/feature_data/mfcc_feature13_{}.xlsx".format(count_t), sheet_name = 'sheet1')


classpreds = np.argmax(predictions, axis=1)  # predicted classes
target = np.argmax(label_test, axis=1)  # true classes
c_names = ['normal','crackles','wheezels','Both']
#print('predictions=', classpreds)
#print('target class=', target)
# Classification Report
print(classification_report(target, classpreds, target_names=c_names))
# Confusion Matrix
print(confusion_matrix(target, classpreds))