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

batch_size = 128
num_epochs = 600
learning_rate = 2e-3
filter_size1 = 10
filter_size2 = 2
dropout = 0.375
classes = 6

mypath2 = "/home/iichsk/workspace/dataset/full_image/audio_and_txt_files3"

filenames2 = [f for f in listdir(mypath2) if (isfile(join(mypath2, f)) and f.endswith('.png'))]

p_id_in_file2 = [] # patient IDs corresponding to each file


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

filepaths2 = [join(mypath2, f) for f in filenames2] # full paths of files

p_diag = pd.read_csv("/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis5.csv",header=None)


labels2 = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file2])

img_features = []

def extract_image(file_name):
    im = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    #im = plt.resize(im, (640, 480))
    #print(im.shape)
    return im
# Iterate through each sound file and extract the features


for file_name in filepaths2:
    data = extract_image(file_name)
    #print("data=",data.shape)
    img_features.append(data)

print('Finished feature extraction from ', len(img_features), ' files')


img_features = np.array(img_features)
#print('img_features', img_features.shape)
img_features1 = np.delete(img_features, np.where((labels2 == 'Asthma') | (labels2 == 'LRTI'))[0], axis=0)
img_labels1 = np.delete(labels2, np.where((labels2 == 'Asthma') | (labels2 == 'LRTI'))[0], axis=0)
#labels2 = np.delete(labels, np.where((labels != 'Healthy'))[0], axis=0)
#features2 = np.delete(features, np.where((labels != 'Healthy'))[0], axis=0)

unique_elements2, counts_elements2 = np.unique(img_labels1, return_counts=True)

print(np.asarray((unique_elements2, counts_elements2)))

le = LabelEncoder()
i_labels2 = le.fit_transform(img_labels1)

#print('i_labels.shape=', i_labels.shape)

oh_labels2 = to_categorical(i_labels2)

#print('spt_features1.shape1=', spt_features1.shape)
img_features1 = np.reshape(img_features1, (*img_features1.shape, 1))
#print('img_features1', img_features1.shape)

#print("features1_reshape=", features1.shape)
img_train, img_test, label_train, label_test = train_test_split(img_features1, oh_labels2, stratify=i_labels2,
                                                    test_size=0.2, random_state = 42)



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

class cnn(nn.Module):

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
# hyper-parameters

def train():

    z = np.random.permutation(img_train.shape[0])

    trn_loss_list = []
    print('train start!!!!!')
    cnn.train()
    for epoch in range(num_epochs):
        trn_loss = 0.0
        for i in range(int(img_train.shape[0] / batch_size)):
            img_input = torch.Tensor(img_train[z[i*batch_size:(i+1)* batch_size], :, :, :]).cuda()

            label =  torch.Tensor(label_train[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            # grad init
            optimizer.zero_grad()
            model_output = cnn(img_input)
            
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
    torch.save(cnn.state_dict(), "/home/iichsk/workspace/weight/iic_img_cnn_weight_{}_{}_{}_{}.pt".format(classes, num_epochs, batch_size, count_t))


def test():
    cnn.eval()
    test_loss = 0.0
    predictions = np.zeros((int(img_test.shape[0]), classes))
    target = np.zeros((int(label_test.shape[0]), classes))
    correct = 0
    wrong = 0
    for j in range(int(img_test.shape[0])):
        #spt_input = torch.Tensor(spt_test[j:(j+1), :, :, :]).cuda()
        img_input = torch.Tensor(img_test[j:(j+1), :, :, :]).cuda()

        test_label =  torch.Tensor(label_test[j:(j+1), :]).cuda()
        test_output = cnn(img_input)

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
c_names = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Pneumonia', 'URTI', 'Healthy']
#print('predictions=', classpreds)
print('target class=', target)
# Classification Report
print(classification_report(target, classpreds, target_names=c_names))
# Confusion Matrix
print(confusion_matrix(target, classpreds))