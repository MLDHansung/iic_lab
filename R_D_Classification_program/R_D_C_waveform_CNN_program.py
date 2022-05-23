import os
from os import listdir
from os.path import isfile, join

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import time
from torchsummary import summary

batch_size = 128
num_epochs = 600
learning_rate = 2e-3
filter_size1 = 10
filter_size2 = 2
dropout = 0.5
classes = 6

path = os.getcwd()
mypath = path+"/database/waveform_image/"
p_diag = pd.read_csv(path+"/database/label/patient_diagnosis.csv",header=None)

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.png'))]
p_id_in_file = [] # patient IDs corresponding to each file
for name in filenames:
    p_id_in_file.append(int(name[:3]))
p_id_in_file = np.array(p_id_in_file)

filepaths = [join(mypath, f) for f in filenames] # full paths of files
labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])

img_features = []
def extract_image(file_name):
    im = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    #im = plt.resize(im, (640, 480))
    #print(im.shape)
    return im
# Iterate through each sound file and extract the features

for file_name in filepaths:
    data = extract_image(file_name)
    img_features.append(data)

print('Finished feature extraction from ', len(img_features), ' files')

img_features = np.array(img_features)
img_features1 = np.delete(img_features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)
labels1 = np.delete(labels, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)


unique_elements, counts_elements = np.unique(labels1, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

le = LabelEncoder()
i_labels = le.fit_transform(labels1)
oh_labels = to_categorical(i_labels)
img_features1 = np.reshape(img_features1, (*img_features1.shape, 1))
img_train, img_test, label_train, label_test = train_test_split(img_features1, oh_labels, stratify=i_labels,
                                                    test_size=0.2, random_state = 42)

img_train = np.transpose(img_train, (0,3,1,2))
img_test = np.transpose(img_test, (0,3,1,2))

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

        return img_x
cnn = cnn()
cnn.cuda()
# backpropagation method
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
cnn.load_state_dict(torch.load(path+"/weight/img_cnn_weight_6_800_128_best.pt"),strict=False)
summary(cnn,(1,240,861))

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

            # 학습과정 출력
        if (epoch + 1) % 20 == 0:  #
            print("epoch: {}/{} | trn loss: {:.5f}".format(
                epoch + 1, num_epochs, trn_loss / 100
            ))
            trn_loss_list.append(trn_loss / 100)
    count_t = time.time()
    print("Model's state_dict:")
    torch.save(cnn.state_dict(), path+"/weight/img_cnn_weight_{}_{}_{}_{}.pt".format(classes, num_epochs, batch_size, count_t))


def test():
    cnn.eval()
    predictions = np.zeros((int(img_test.shape[0]), classes))
    target = np.zeros((int(label_test.shape[0]), classes))
    
    for j in range(int(img_test.shape[0])):
        img_input = torch.Tensor(img_test[j:(j+1), :, :, :]).cuda()
        test_label =  torch.Tensor(label_test[j:(j+1), :]).cuda()
        test_output = cnn(img_input)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        target[j:j+1, :] = test_label.detach().cpu().numpy()

    return predictions, target

# train()
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