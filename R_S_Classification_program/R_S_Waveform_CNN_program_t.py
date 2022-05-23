import os
from os import listdir
from os.path import isfile, join
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import time

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
from torchsummary import summary

path = os.getcwd()
mypath = path+"/database/audio_files/"
mypath_waveform = path+"/database/waveform_image/"
p_diag = pd.read_csv(path+"/database/label/patient_diagnosis_sliced.csv",header=None) #patient diagnosis
filenames_waveform = [f for f in listdir(mypath_waveform) if (isfile(join(mypath_waveform, f)) and f.endswith('.png'))]
p_id_in_file_waveform = [] # patient IDs corresponding to each file

for name in filenames_waveform:
    p_id_in_file_waveform.append(int(name.split('_')[0]))
p_id_in_file_waveform = np.array(p_id_in_file_waveform)

def extract_image(file_name):
    im = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    return im

filepaths_waveform = [join(mypath_waveform, f) for f in filenames_waveform] # full paths of files
labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file_waveform])

img_features = []
for file_name2 in filepaths_waveform:
    data = extract_image(file_name2)
    img_features.append(data)


print('Finished feature extraction from ', len(img_features), ' files')

img_features = np.array(img_features)
img_features1 = np.delete(img_features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)
labels1 = np.delete(labels, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)

le = LabelEncoder()
i_labels = le.fit_transform(labels1)
oh_labels = to_categorical(i_labels)

img_features1 = np.reshape(img_features1, (*img_features1.shape, 1))

waveform_dataset = np.transpose(img_features1, (0,3,1,2))

waveform_train, waveform_test, label_train, label_test = train_test_split(waveform_dataset, oh_labels, stratify=i_labels,
                                                    test_size=0.2, random_state = 42)
print('waveform_train.shape',waveform_train.shape)

# construct model on cuda if available
use_cuda = torch.cuda.is_available()
# loss
criterion = nn.MultiLabelSoftMarginLoss()

class img_cnn(nn.Module): # Waveform CNN Class

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
img_cnn.load_state_dict(torch.load(path+"/weight/symptom_wav_cnn_weight_4_1000_128_1626951295.9009268.pt"))
img_optimizer = optim.Adam(img_cnn.parameters(), lr=learning_rate)
summary(img_cnn,(1, 240, 861))

def img_train(img_trainset,feature_label):
    num_epochs =  350
    z = np.random.permutation(img_trainset.shape[0])
    trn_loss_list = []
    print('img train start!!!!!')
    img_cnn.train()
    for epoch in range(num_epochs):
        trn_loss = 0.0
        for i in range(int(img_trainset.shape[0] / batch_size)):
            img_input = torch.Tensor(img_trainset[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            label =  torch.Tensor(feature_label[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            
            # grad init
            img_optimizer.zero_grad()
            model_output = img_cnn(img_input)
            # calculate loss
            img_loss = criterion(model_output, label)
            # back propagation
            img_loss.backward(retain_graph=True)
            # weight update
            img_optimizer.step()
            # trn_loss summary
            trn_loss += img_loss.item()            
            # 학습과정 출력
        if (epoch + 1) % 50 == 0:  #
            print("epoch: {}/{} | trn loss: {:.8f}".format(
                epoch + 1, num_epochs, trn_loss / 100
            ))
            trn_loss_list.append(trn_loss / 100)
    count_t = time.time()
    torch.save(img_cnn.state_dict(), path+"/weight/waveform_CNN_weight_{}_{}_{}.pt".format(num_epochs, batch_size, count_t))

def img_cnn_test(img_data): # Waveform CNN test
    img_cnn.eval()
    predictions = np.zeros((int(img_data.shape[0]), classes))
    img_feature = np.zeros((int(img_data.shape[0]), 128))

    for j in range(int(img_data.shape[0])):
        img_input = torch.Tensor(img_data[j:(j+1), :, :, :]).cuda()
        test_output, img_feature_x = img_cnn(img_input)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        img_feature[j:j+1,:] = img_feature_x.data.cpu().numpy()

    return predictions, img_feature

## CNN model's feature
predictions, _ = img_cnn_test(waveform_test)

## test ensemble model
target = np.argmax(label_test, axis=1)  # true classes
classpreds = np.argmax(predictions, axis=1)  # predicted classes
c_names = ['normal','wheezes','crackles','Both']
## Classification Report
print('#'*10,'Triple Ensemble CNN report','#'*10)
print(classification_report(target, classpreds, target_names=c_names))
## Confusion Matrix
print(confusion_matrix(target, classpreds))

