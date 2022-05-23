from datetime import datetime
import os
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
from res_slide import file_Slide
file_Slide=file_Slide()

batch_size = 128
num_epochs =300
learning_rate = 1e-3
filter_size1 = 3
filter_size2 = 3
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
    return norm_log_S
                        



filepaths =[s.split('.')[0] for s in os.listdir(mypath) if '.wav' in s] # full paths of files

p_diag = pd.read_csv("/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/csv_data/data2.csv",header=None)


labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])

spt_features = []
print('filepaths',len(filepaths))
for file_name in filepaths:
    #data = extract_features(file_name)
    #print("data=",data[1].shape)
    #print(int(file_name.split('_')[0]))

    #spt_features.append(int(file_name.split('_')[0]))
    #print(spt_features)
    spt_features.append(file_name)
    #size_features.append((data[1].shape))

print('Finished feature extraction from ', len(spt_features), ' files')

spt_features = np.array(spt_features)
#spt_features1 = np.delete(spt_features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)
#labels1 = np.delete(labels, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)

unique_elements, counts_elements = np.unique(labels, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

le = LabelEncoder()
i_labels = le.fit_transform(labels)
oh_labels = to_categorical(i_labels)
#spt_features1 = np.reshape(spt_features1, (*spt_features1.shape, 1))
#print('spt_features',spt_features)
#spt_features=np.squeeze(spt_features)
print('spt_features',spt_features.shape)

spt_train, spt_test, label_train, label_test = train_test_split(spt_features, oh_labels, stratify=i_labels,
                                                    test_size=0.2, random_state = 42)
print('spt_train',spt_train.shape)
print('label_train',label_train.shape)
#spt_train = np.transpose(spt_train, (0,3,1,2))
#spt_test = np.transpose(spt_test, (0,3,1,2))

# construct model on cuda if available
use_cuda = torch.cuda.is_available()

# loss
criterion = nn.MultiLabelSoftMarginLoss()

class spt_cnn(nn.Module):

    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(spt_cnn, self).__init__()
        self.spt_bn1= torch.nn.BatchNorm2d(1)
        self.spt_conv1 = nn.Conv2d(1, 64, kernel_size=filter_size1)  # 16@39*861
        self.spt_dropout1 = nn.Dropout(dropout)

        self.spt_bn2= torch.nn.BatchNorm2d(64)
        self.spt_conv2 = nn.Conv2d(64, 128, kernel_size=filter_size2)  # 32@18*429
        
        self.spt_bn3= torch.nn.BatchNorm2d(128)
        self.spt_conv3 = nn.Conv2d(128, 128, kernel_size=filter_size2)  # 64@8*213
        
        self.spt_bn4= torch.nn.BatchNorm2d(128)
        self.spt_conv4 = nn.Conv2d(128, 256, kernel_size=filter_size2)  # 128@3*105
        
        # self.spt_bn5= torch.nn.BatchNorm2d(256)
        # self.spt_conv5 = nn.Conv2d(256, 256, kernel_size=filter_size2)  # 128@3*105

        # self.spt_bn6= torch.nn.BatchNorm2d(256)
        # self.spt_conv6 = nn.Conv2d(256, 512, kernel_size=filter_size2)  # 128@3*105
        
        # self.spt_bn7= torch.nn.BatchNorm2d(512)       
        # self.spt_conv7 = nn.Conv2d(512, 512, kernel_size=filter_size2)  # 128@3*105

        # self.spt_bn8= torch.nn.BatchNorm2d(512)       
        self.spt_global_pool = nn.AdaptiveAvgPool2d(1)

        self.spt_fc1 = nn.Linear(256, 256)
        self.spt_bn9 = torch.nn.BatchNorm1d(256)

        self.spt_fc2 = nn.Linear(256, classes) 
        
        self.spt_relu = nn.ReLU()
    
    def forward(self, spt_x):
        #print(spt_x.shape)
        spt_x = self.spt_bn1(spt_x)
        spt_x = self.spt_relu(self.spt_conv1(spt_x))
        spt_x = self.spt_dropout1(spt_x)

        spt_x = self.spt_bn2(spt_x)
        spt_x = self.spt_relu(self.spt_conv2(spt_x))
        spt_x = self.spt_dropout1(spt_x)

        spt_x = self.spt_bn3(spt_x)
        spt_x = self.spt_relu(self.spt_conv3(spt_x))
        spt_x = self.spt_dropout1(spt_x)
        
        spt_x = self.spt_bn4(spt_x)
        spt_x = self.spt_relu(self.spt_conv4(spt_x))
        spt_x = self.spt_dropout1(spt_x)
        
        # spt_x = self.spt_bn5(spt_x)
        # spt_x = self.spt_relu(self.spt_conv5(spt_x))
        # spt_x = self.spt_dropout1(spt_x)

        # spt_x = self.spt_bn6(spt_x)
        # spt_x = self.spt_relu(self.spt_conv6(spt_x))
        # spt_x = self.spt_dropout1(spt_x)

        # spt_x = self.spt_bn7(spt_x)
        # spt_x = self.spt_relu(self.spt_conv7(spt_x))

        # spt_x = self.spt_bn8(spt_x)
        spt_x = self.spt_global_pool(spt_x) # batchsize x netdim x 1 x 1
        
        spt_feature_x = spt_x
        spt_x = spt_x.view(spt_x.size(0), -1) # batchsize x netdim 

        spt_x = self.spt_fc1(spt_x) 
        spt_x = self.spt_relu(spt_x)
        spt_x = self.spt_bn9(spt_x)
        spt_x = self.spt_fc2(spt_x)

        return spt_x, spt_feature_x

spt_cnn = spt_cnn()
spt_cnn.cuda()
optimizer = optim.Adam(spt_cnn.parameters(), lr=learning_rate)
spt_cnn.load_state_dict(torch.load("/home/iichsk/workspace/wavpreprocess/weight/symptom_spt_cnn_weight_4_300_128_1625474397.8184922.pt"))


def train():
    #z = np.random.permutation(spt_train.shape[0]*2)
    trn_loss_list = []
    print('train start!!!!!')
    spt_cnn.train()
    spt_input1_feature = np.zeros((int(spt_train.shape[0]),40,100))
    spt_input2_feature = np.zeros((int(spt_train.shape[0]),40,100))
    spt_input3_feature = np.zeros((int(spt_train.shape[0]),40,100))

    for epoch in range(num_epochs):
        
        trn_loss = 0.0
        for i in range(int(spt_train.shape[0] / batch_size)):
            for ii in range(int(batch_size)):
                #print('i=',i,'ii=',ii)
                
                path= mypath + spt_train[(batch_size*i)+ii]
                # print('path=',path)
                feature_data1, feature_data2, feature_data3 = file_Slide.getSlindingfile(path,spt_train[(batch_size*i)+ii])
                spt_input1_feature[(batch_size*i)+ii]=(feature_data1)
                spt_input2_feature[(batch_size*i)+ii]=(feature_data2)
                spt_input3_feature[(batch_size*i)+ii]=(feature_data3)
                   
            spt_input1 = np.reshape(spt_input1_feature, (*spt_input1_feature.shape, 1))     
            spt_input1 = np.transpose(spt_input1, (0,3,1,2))
            spt_input2 = np.reshape(spt_input2_feature, (*spt_input2_feature.shape, 1))     
            spt_input2 = np.transpose(spt_input2, (0,3,1,2))
            spt_input3 = np.reshape(spt_input3_feature, (*spt_input3_feature.shape, 1))     
            spt_input3 = np.transpose(spt_input3, (0,3,1,2))
            spt_input_con = np.concatenate((spt_input1,spt_input2,spt_input3),axis=0)

            spt_input = torch.Tensor(spt_input_con[i*batch_size*2:(i+1)* batch_size*2-1, :, :, :]).cuda()

            label_train_con = np.concatenate((label_train,label_train),axis=0)

            label =  torch.Tensor(label_train_con[i*batch_size*2:(i+1)* batch_size*2-1, :]).cuda()
            # grad init
            optimizer.zero_grad()
            model_output, _ = spt_cnn(spt_input)
            del spt_input
            # calculate loss
            loss = criterion(model_output, label)
            # back propagation
            loss.backward()
            # weight update
            optimizer.step()

            # trn_loss summary
            trn_loss += loss.item()
           # del (memory issue)
            del loss
            del model_output

            # 학습과정 출력
        if (epoch + 1) % 1 == 0:  #
            print("epoch: {}/{} | trn loss: {:.5f}".format(
                epoch + 1, num_epochs, trn_loss / 100
            ))
            trn_loss_list.append(trn_loss / 100)
    count_t = time.time()
    print("Model's state_dict:")
    torch.save(spt_cnn.state_dict(), "/home/iichsk/workspace/wavpreprocess/weight/symptom_spt_cnn_weight_{}_{}_{}_{}.pt".format(classes, num_epochs, batch_size, count_t))


def test():
    spt_cnn.eval()
    test_loss = 0.0
    spt_input1_feature = np.zeros((int(spt_test.shape[0]),40,100))
    spt_input2_feature = np.zeros((int(spt_test.shape[0]),40,100))
    spt_input3_feature = np.zeros((int(spt_test.shape[0]),40,100))

    predictions = np.zeros((int(spt_test.shape[0]), classes))

    target = np.zeros((int(label_test.shape[0]), classes))
    correct = 0
    wrong = 0
    for i in range(int(spt_test.shape[0])):
        
            
        path= mypath + spt_train[i]
        feature_data1, feature_data2, feature_data3 = file_Slide.getSlindingfile(path,spt_train[i])
        spt_input1_feature[i]=(feature_data1)
        spt_input2_feature[i]=(feature_data2)
        spt_input3_feature[i]=(feature_data3)
                
        spt_input1 = np.reshape(spt_input1_feature, (*spt_input1_feature.shape, 1))     
        spt_input1 = np.transpose(spt_input1, (0,3,1,2))
        spt_input2 = np.reshape(spt_input2_feature, (*spt_input2_feature.shape, 1))     
        spt_input2 = np.transpose(spt_input2, (0,3,1,2))
        spt_input3 = np.reshape(spt_input3_feature, (*spt_input3_feature.shape, 1))     
        spt_input3 = np.transpose(spt_input3, (0,3,1,2))  

        spt_input1 = torch.Tensor(spt_input1[i:(i+1), :, :, :]).cuda()
        spt_input2 = torch.Tensor(spt_input2[i:(i+1), :, :, :]).cuda()
        spt_input3 = torch.Tensor(spt_input3[i:(i+1), :, :, :]).cuda()

        test_label =  torch.Tensor(label_test[i:(i+1), :]).cuda()
        test_output1, _ = spt_cnn(spt_input1)
        test_output2, _ = spt_cnn(spt_input2)
        test_output3, _ = spt_cnn(spt_input3)

        print('test_output1=',test_output1)
        print('test_output2=',test_output2)
        print('test_output3=',test_output3)

        test_output1=test_output1.detach().cpu().numpy()
        test_output2=test_output2.detach().cpu().numpy()
        test_output3=test_output3.detach().cpu().numpy()

        test_output1_argmax=np.argmax(test_output1)
        test_output2_argmax=np.argmax(test_output2)
        test_output3_argmax=np.argmax(test_output3)

        print('test_output1_argmax=',test_output1_argmax)
        print('test_output2_argmax=',test_output2_argmax)
        print('test_output3_argmax=',test_output3_argmax)

        if (test_output1_argmax == 0) & (test_output2_argmax == 0) & (test_output3_argmax == 0):
            test_output_vote_result = np.array([1,0,0,0])
        else:
            if (test_output1_argmax == 0):
                if(test_output2_argmax == 0):
                    test_output_vote_result = test_output3
                elif(test_output3_argmax == 0):
                    test_output_vote_result = test_output2
                else:
                    if (test_output2_argmax != test_output3_argmax):
                        test_output_vote_result = test_output2 + test_output3
                    else:
                        test_output_vote_result = test_output2
            elif(test_output2_argmax == 0):
                if(test_output3_argmax == 0):
                    test_output_vote_result = test_output1
                else:
                    if (test_output1_argmax != test_output3_argmax):
                        test_output_vote_result = test_output1 + test_output3
                    else:
                        test_output_vote_result = test_output1
            elif(test_output3_argmax == 0):
                if (test_output1_argmax != test_output2_argmax):
                    test_output_vote_result = test_output1 + test_output2
                else:
                    test_output_vote_result = test_output1
            else:
                test_output_vote_result = test_output1 + test_output2 + test_output3
       
        print('test_output_vote_result=',test_output_vote_result)
        predictions[i:i+1,:] = test_output_vote_result
        print('label=',test_label)
        target[i:i+1, :] = test_label.detach().cpu().numpy()
        
        

    #print(" Error: {}/{} ({:.2f}%)".format(wrong,wrong+correct, 100.*(float(wrong)/float(correct+wrong))))
    #print("eval_prediction.shape:", predictions.shape)
    return predictions, target

#train()
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

spt1_classpreds=np.expand_dims(classpreds, axis=1)
target=np.expand_dims(target, axis=1)

#print('spt1_classpreds.shape',spt1_classpreds.shape)
print('target.shape',target.shape)
concat_pred = np.concatenate((spt1_classpreds, target),1)

# df = pd.DataFrame(concat_pred,columns=['spt1','target'])
# print(df)
# count_t = time.time()
# df.to_excel("/home/iichsk/Desktop/share/workspace/wavpreprocess/predict_value_{}.xlsx".format(count_t), sheet_name = 'sheet1')
