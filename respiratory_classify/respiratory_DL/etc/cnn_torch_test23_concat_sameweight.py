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
import torchvision

import cv2
from torch.autograd import Variable
import time


batch_size = 128
learning_rate = 1e-3
filter_size1 = 10
filter_size2 = 2
classes = 6
spt_dropout = 0.325
img_dropout = 0.325
mypath = "/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files2"
mypath2 = "/home/iichsk/workspace/dataset/full_image/audio_and_txt_files2"

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]
filenames2 = [f for f in listdir(mypath2) if (isfile(join(mypath2, f)) and f.endswith('.png'))]

p_id_in_file = [] # patient IDs corresponding to each file
p_id_in_file2 = [] # patient IDs corresponding to each file

for name in filenames:
    p_id_in_file.append(int(name[:3]))

p_id_in_file = np.array(p_id_in_file)

for name in filenames2:
    p_id_in_file2.append(int(name[:3]))

p_id_in_file2 = np.array(p_id_in_file2)

max_pad_len = 862 # to make the length of all MFCC equal

def extract_features(file_name):
    """
    This function takes in the path for an audio file as a string, loads it, and returns the MFCC
    of the audio"""

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

filepaths = [join(mypath, f) for f in filenames] # full paths of files
filepaths2 = [join(mypath2, f) for f in filenames2] # full paths of files

p_diag = pd.read_csv("/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv",header=None)


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

for file_name in filepaths2:
    data = extract_image(file_name)
    #print("data=",data.shape)
    img_features.append(data)

print('Finished feature extraction from ', type(spt_features), ' files')
print('Finished feature extraction from ', len(img_features), ' files')

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
img_test26 = np.squeeze(img_test)
print('img_test26',img_test26.shape)
img_test26=np.squeeze(img_test26[26:27,:,:])
print('img_test26',img_test26.shape)
spt_test26=np.squeeze(spt_test)
spt_test26=np.squeeze(spt_test26[26:27,:,:])
print('spt_test26',spt_test26.shape)
plt.imshow(img_test26)
plt.savefig('/home/iichsk/workspace/wavpreprocess/errorfigure/WAVE_26_pneumonia.png') 
plt.close()
librosa.display.specshow(spt_test26)
plt.savefig('/home/iichsk/workspace/wavpreprocess/errorfigure/MFCC_26_pneumonia.png')

spt_train = np.transpose(spt_train, (0,3,1,2))
spt_test = np.transpose(spt_test, (0,3,1,2))
img_train = np.transpose(img_train, (0,3,1,2))
img_test = np.transpose(img_test, (0,3,1,2))

#print('img_test=', label2_test)



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
        spt_feature_x = spt_x
        spt_x = spt_x.view(spt_x.size(0), -1) # batchsize x netdim 

        spt_x = self.spt_fc1(spt_x) 
        spt_x = self.spt_bn1(spt_x)
        spt_x = self.spt_fc2(spt_x)

        return spt_x, spt_feature_x

spt_cnn = spt_cnn()
spt_cnn.cuda()
spt_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/spt_cnn_weight_6_600_128_best.pt"))
spt_optimizer = optim.Adam(spt_cnn.parameters(), lr=learning_rate)

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
        self.img_fc1 = nn.Linear(256, 128)
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
        img_feature_x = img_x

        img_x = self.img_fc2(img_x)  #

        return img_x, img_feature_x
img_cnn = img_cnn()
img_cnn.cuda()
img_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/img_cnn_weight_6_800_128_best.pt"))

img_optimizer = optim.Adam(img_cnn.parameters(), lr=learning_rate)


class concat_fc(nn.Module):
    def __init__(self):
        super(concat_fc, self).__init__()

       
        self.concat_fc1 = nn.Linear(256, 128)
        self.concat_fc2 = nn.Linear(128, classes)


        self.concat_relu = nn.ReLU()
        self.concat_bn1 = torch.nn.BatchNorm1d(128)


        self.concat_dropout1 = nn.Dropout(0.4)


       
    
    def forward(self, concat_x):
    
        
        #concat_x = concat_x.view(concat_x.size(0), -1)
        concat_x = self.concat_fc1(concat_x)  #
        concat_x = self.concat_bn1(concat_x)
        concat_x = self.concat_dropout1(concat_x)

        concat_x = self.concat_fc2(concat_x)  #
        

        return concat_x


concat_fc = concat_fc()
concat_fc.cuda()
#print('model-------', cnn)
# backpropagation method
concat_optimizer = optim.Adam(concat_fc.parameters(), lr=learning_rate)
z = np.random.permutation(spt_train.shape[0])



def concat_train(concat_trainset,feature_label):
    num_epochs = 450
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
    spt_feature = np.zeros((int(spt_data.shape[0]), 128, 1,1))
    #feature_label = np.zeros((spt_data.shape[0], 6))

    #target = np.zeros((int(label.shape[0]), classes))
    correct = 0
    wrong = 0
    for j in range(int(spt_data.shape[0])):
        #spt_input = torch.Tensor(spt_test[j:(j+1), :, :, :]).cuda()
        spt_input = torch.Tensor(spt_data[j:(j+1), :, :, :]).cuda()

        #test_label =  torch.Tensor(label[j:(j+1), :]).cuda()
        #feature_label[j:(j+1),:] = test_label.data.cpu()

        test_output, spt_feature_x = spt_cnn(spt_input)
        #print('spt_feature_x.shape',spt_feature_x.shape)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        spt_feature[j:j+1,:,:,:] = spt_feature_x.data.cpu().numpy()

        '''target[j:j+1, :] = test_label.detach().cpu().numpy()
                                pred = test_output.data.max(1)[1]
                                one_hot_pred = torch.zeros(1, classes)
                                one_hot_pred[range(one_hot_pred.shape[0]), pred]=1
                                correct += one_hot_pred[0, 0].eq(test_label.data[0, 0].cpu()).cpu().sum()
                                wrong +=  len(test_label)-one_hot_pred[0, 0].eq(test_label.data[0, 0].cpu()).cpu().sum()'''


    #print(" Error: {}/{} ({:.2f}%)".format(wrong,wrong+correct, 100.*(float(wrong)/float(correct+wrong))))
    #print("eval_prediction.shape:", predictions.shape)
    return predictions, spt_feature

def img_cnn_test(img_data):
    img_cnn.eval()
    test_loss = 0.0
    predictions = np.zeros((int(img_data.shape[0]), classes))
    img_feature = np.zeros((int(img_data.shape[0]), 128))
    #target = np.zeros((int(img_data.shape[0]), classes))
    correct = 0
    wrong = 0
    for j in range(int(img_data.shape[0])):
        #spt_input = torch.Tensor(spt_test[j:(j+1), :, :, :]).cuda()
        img_input = torch.Tensor(img_data[j:(j+1), :, :, :]).cuda()

        #test_label =  torch.Tensor(label[j:(j+1), :]).cuda()
        test_output, img_feature_x = img_cnn(img_input)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        img_feature[j:j+1,:] = img_feature_x.data.cpu().numpy()

        '''target[j:j+1, :] = test_label.detach().cpu().numpy()
                                pred = test_output.data.max(1)[1]
                                one_hot_pred = torch.zeros(1, classes)
                                one_hot_pred[range(one_hot_pred.shape[0]), pred]=1
                                correct += one_hot_pred[0, 0].eq(test_label.data[0, 0].cpu()).cpu().sum()
                                wrong +=  len(test_label)-one_hot_pred[0, 0].eq(test_label.data[0, 0].cpu()).cpu().sum()'''


    #print(" Error: {}/{} ({:.2f}%)".format(wrong,wrong+correct, 100.*(float(wrong)/float(correct+wrong))))
    #print("eval_prediction.shape:", predictions.shape)
    return predictions, img_feature

def concat_test(concat_testset):
    concat_fc.eval()
    test_loss = 0.0
    predictions = np.zeros((int(concat_testset.shape[0]), classes))

    #target = np.zeros((int(test_feature_label.shape[0]), 6))
    correct = 0
    wrong = 0
    for j in range(int(concat_testset.shape[0])):
        #spt_input = torch.Tensor(spt_test[j:(j+1), :, :, :]).cuda()
        concat_input = torch.Tensor(concat_testset[j:(j+1), :]).cuda()
        #test_label =  torch.Tensor(test_feature_label[j:(j+1), :]).cuda()
        test_output = concat_fc(concat_input)

        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        #target[j:j+1, :] = test_label.detach().cpu().numpy()
        #pred = test_output.data.max(1)[1]
        #one_hot_pred = torch.zeros(1, classes)
        #one_hot_pred[range(one_hot_pred.shape[0]), pred]=1
        #correct += one_hot_pred[0, 0].eq(test_label.data[0, 0].cpu()).cpu().sum()
        #wrong +=  len(test_label)-one_hot_pred[0, 0].eq(test_label.data[0, 0].cpu()).cpu().sum()


    #print(" Error: {}/{} ({:.2f}%)".format(wrong,wrong+correct, 100.*(float(wrong)/float(correct+wrong))))
    #print("eval_prediction.shape:", predictions.shape)
    return predictions


#spectrum part
_, spt_feature_x = spt_cnn_test(spt_train)
print(spt_feature_x.shape)
spt_feature_x = np.squeeze(spt_feature_x)
spt_prediction, spt_feature = spt_cnn_test(spt_test)
spt_feature = np.squeeze(spt_feature)
print(spt_feature_x.shape)
classpreds = np.argmax(spt_prediction, axis=1)  # predicted classes
target = np.argmax(label_test, axis=1)  # true classes
c_names = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Pneumonia', 'URTI', 'Healthy']
# Classification Report
print(classification_report(target, classpreds, target_names=c_names))
# Confusion Matrix
print(confusion_matrix(target, classpreds))
print('MFCC classprediction= ')
print(classpreds)

#waveform part
_, img_feature_x = img_cnn_test(img_train)
img_prediction, img_feature = img_cnn_test(img_test)
classpreds = np.argmax(img_prediction, axis=1)  # predicted classes
target = np.argmax(label_test, axis=1)  # true classes
c_names = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Pneumonia', 'URTI', 'Healthy']
# Classification Report
print(classification_report(target, classpreds, target_names=c_names))
# Confusion Matrix
print(confusion_matrix(target, classpreds))
print('waveform classprediction= ')
print(classpreds)
print('spt_feature_x',spt_feature_x.shape)
print('img_feature_x',img_feature_x.shape)
#concat data part
concat_trainset = np.concatenate((spt_feature_x, img_feature_x),1)
concat_train(concat_trainset,label_train)

concat_testset = np.concatenate((spt_feature, img_feature),1)
predictions = concat_test(concat_testset)

classpreds = np.argmax(predictions, axis=1)  # predicted classes
print('classpreds', classpreds)
target = np.argmax(label_test, axis=1)  # true classes
print('target', target)
#c_names = ['Abnormal', 'Healthy']
c_names = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Pneumonia', 'URTI', 'Healthy']

# Classification Report
print(classification_report(target, classpreds, target_names=c_names))
# Confusion Matrix
print(confusion_matrix(target, classpreds))


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