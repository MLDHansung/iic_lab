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
from matplotlib import cm
from sklearn.manifold import TSNE
from tsne import bh_sne
from mpl_toolkits.mplot3d import Axes3D

batch_size = 512
learning_rate = 1e-3
classes = 2
MFCC_dropout = 0.3
mfcc_num_epochs = 1000

mfcc_testset = np.load('mfcc_testset_normal.npy')
test_label = np.load('test_label_normal.npy')
trainset_p_id, mfcc_testset, train_label, test_label = train_test_split(mfcc_testset, test_label, stratify=test_label,
                                                    test_size=0.2, random_state = 42)
use_cuda = torch.cuda.is_available()

# loss
criterion = nn.MultiLabelSoftMarginLoss()
# MFCC CNN model
class MFCC_cnn(nn.Module):

    def __init__(self):

        super(MFCC_cnn, self).__init__()

        self.MFCC_conv1 = nn.Conv2d(1, 16, kernel_size=5)  # 
        self.MFCC_bn1 = torch.nn.BatchNorm2d(16)
        self.MFCC_pool1 = nn.MaxPool2d(2)  #
        self.MFCC_conv2 = nn.Conv2d(16, 32, kernel_size=3)  # 
        self.MFCC_bn2 = torch.nn.BatchNorm2d(32)
        self.MFCC_pool2 = nn.MaxPool2d(2)  # 
        self.MFCC_dropout2 = nn.Dropout(MFCC_dropout)
        self.MFCC_conv3 = nn.Conv2d(32, 64, kernel_size=(1,2))  # 
        self.MFCC_bn3 = torch.nn.BatchNorm2d(64)
        self.MFCC_pool3 = nn.MaxPool2d(2)  # 
        self.MFCC_dropout3 = nn.Dropout(MFCC_dropout)
        self.MFCC_conv4 = nn.Conv2d(64, 128, kernel_size=(1,2))  # 
        self.MFCC_bn4 = torch.nn.BatchNorm2d(128)
        self.MFCC_pool4 = nn.MaxPool2d(2)  # 
        self.MFCC_conv5 = nn.Conv2d(128, 256, kernel_size=(1,2))  # 
        self.MFCC_bn5 = torch.nn.BatchNorm2d(256)
        self.MFCC_pool5 = nn.MaxPool2d(2)  # 


        self.MFCC_global_pool = nn.AdaptiveAvgPool2d(1)
        # self.MFCC_bn0 = torch.nn.BatchNorm2d(128)
        self.MFCC_fc1 = nn.Linear(256, 128)
        self.MFCC_fc_bn1 = torch.nn.BatchNorm1d(128)
        self.MFCC_fc2 = nn.Linear(128, classes)  
        self.MFCC_relu = nn.ReLU()
    
    def forward(self, MFCC_x):

        MFCC_x = self.MFCC_relu(self.MFCC_bn1(self.MFCC_conv1(MFCC_x)))
        MFCC_x = self.MFCC_pool1(MFCC_x)  #
        MFCC_x = self.MFCC_relu(self.MFCC_bn2(self.MFCC_conv2(MFCC_x)))
        MFCC_x = self.MFCC_pool2(MFCC_x)  #
        MFCC_x = self.MFCC_relu(self.MFCC_bn3(self.MFCC_conv3(MFCC_x)))
        MFCC_x = self.MFCC_pool3(MFCC_x)  #
        MFCC_x = self.MFCC_relu(self.MFCC_bn4(self.MFCC_conv4(MFCC_x)))
        MFCC_x = self.MFCC_pool4(MFCC_x)  #
        MFCC_x = self.MFCC_relu(self.MFCC_bn5(self.MFCC_conv5(MFCC_x)))
        MFCC_x = self.MFCC_pool5(MFCC_x)  #

        MFCC_x = self.MFCC_global_pool(MFCC_x)
        MFCC_x = MFCC_x.view(MFCC_x.size(0), -1) # 
        MFCC_x = self.MFCC_fc1(MFCC_x) 
        MFCC_feature_raw = MFCC_x
        MFCC_x = self.MFCC_fc_bn1(MFCC_x)
        MFCC_feature_x = MFCC_x

        MFCC_x = self.MFCC_fc2(MFCC_x)

        return MFCC_x, MFCC_feature_x, MFCC_feature_raw


MFCC_cnn = MFCC_cnn()
MFCC_cnn.cuda()
MFCC_optimizer = optim.Adam(MFCC_cnn.parameters(), lr=learning_rate)
MFCC_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/MFCC_cnn2_normal_weight_200_512_1637402612.2998834.pt"))

# MFCC CNN train
def MFCC_train(mfcc_trainset, label):
    z = np.random.permutation(mfcc_trainset.shape[0])
    trn_loss_list = []
    print('MFCC CNN train start!!!!!')
    MFCC_cnn.train()
    for epoch in range(mfcc_num_epochs):
        trn_loss = 0.0
        for i in range(int(mfcc_trainset.shape[0] / batch_size)):
            mfcc_input = torch.Tensor(mfcc_trainset[z[i*batch_size:(i+1)* batch_size], :, :, :]).cuda()
            label_torch =  torch.Tensor(label[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            # grad init
            MFCC_optimizer.zero_grad()
            MFCC_cnn_output, _, _ = MFCC_cnn(mfcc_input)
            # calculate loss
            loss = criterion(MFCC_cnn_output, label_torch)
            # back propagation
            loss.backward()
            # weight update
            MFCC_optimizer.step()
            # trn_loss summary
            trn_loss += loss.item()
        # 학습과정 출력
        if (epoch + 1) % 50 == 0:  #
            print("epoch: {}/{} | trn loss: {:.5f}".format(
                epoch + 1, mfcc_num_epochs, trn_loss / 100))
            trn_loss_list.append(trn_loss / 100)
    count_t = time.time()
    torch.save(MFCC_cnn.state_dict(), "/home/iichsk/workspace/weight/MFCC_cnn2_weight_{}_{}_{}.pt".format(mfcc_num_epochs, batch_size, count_t))
    print("MFCC CNN Model's state_dict saved")

# MFCC CNN test
def MFCC_cnn_test(mfcc_testset, label):
    MFCC_cnn.eval()
    test_loss = 0.0
    predictions = np.zeros((int(mfcc_testset.shape[0]), classes))
    mfcc_features = np.zeros((int(mfcc_testset.shape[0]), 128))
    mfcc_features_raw = np.zeros((int(mfcc_testset.shape[0]), 128))
    # test_embeddings = torch.zeros((0, 1310), dtype=torch.float32)
    for j in range(int(mfcc_testset.shape[0])):
        mfcc_input = torch.Tensor(mfcc_testset[j:(j+1), :, :, :]).cuda()
        test_output, mfcc_feature, mfcc_feature_raw = MFCC_cnn(mfcc_input)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        mfcc_features[j:j+1,:] = mfcc_feature.data.cpu().numpy()
        mfcc_features_raw[j:j+1,:] = mfcc_feature_raw.data.cpu().numpy()
        # test_embeddings = torch.cat((test_embeddings,mfcc_feature.detach().cpu()),0)
    return predictions, mfcc_features, mfcc_features_raw


# MFCC CNN part
MFCC_predictions, MFCC_test_features, MFCC_features_raw = MFCC_cnn_test(mfcc_testset, test_label)
count_t = time.time()

# CNN models report
MFCC_predictions_arg = np.argmax(MFCC_predictions, axis=1)  
target = np.argmax(test_label, axis=1)  # true classes
target=np.expand_dims(target, axis=1)
### 2d tsne
output_2d = bh_sne(MFCC_predictions)
plt.rcParams['figure.figsize'] = 20, 20
plt.scatter(output_2d[:, 0], output_2d[:, 1], c=target)
plt.savefig('output_2d_all.png', bbox_inches='tight')


MFCC_predictions_arg=np.expand_dims(MFCC_predictions_arg, axis=1)

result_for_excell = np.concatenate((MFCC_predictions_arg,target),1)
df = pd.DataFrame(result_for_excell,columns=['MFCC','target'])
count_t = time.time()
df.to_excel("/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/5classes/MFCC_result{}.xlsx".format(count_t), sheet_name = 'sheet1')

tsne = TSNE(3, verbose=1)
embedded  = tsne.fit_transform(MFCC_predictions)
x_min, x_max = np.min(embedded, 0), np.max(embedded, 0)
embedded = embedded / (x_max - x_min)

fig = plt.figure()
ax = Axes3D(fig)

group = target
cdict = {0:'red',1:'blue',2:'purple',3:'green',4:'lightskyblue',5:'black'}
mdict = {0:'o',1:'v'}

labeldict= {0:'h_normal',1:'d_normal'}


for g in np.unique(group):
    i=np.where(group == g)
    ax.scatter(embedded[i,0],embedded[i,1],embedded[i,2],s=30,marker=mdict[g],label=labeldict[g])
ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
for x in range(0,361,10):
    for y in range(0,361,10):
        ax.view_init(x,y)
        plt.savefig('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/5classes/T-SNE_3D_image_normal_H_D/MFCC_feature_T_sne_{}_{}.png'.format(x,y),bbox_inches='tight')
