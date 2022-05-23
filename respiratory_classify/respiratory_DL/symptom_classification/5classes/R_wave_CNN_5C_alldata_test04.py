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
classes = 5
dropout = 0.3
wave_num_epochs = 100

# dataset load
# mypath = "/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/waveform_image/" # respiratory sound directory
# p_diag = pd.read_csv("/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/csv_data/symptom_label_5classes.csv",header=None) # patient diagnosis csv file

# filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.png'))]

# p_id_in_file = [] # patient IDs corresponding to each file
# for name in filenames:
#     p_id_in_file.append(int(name.split('_')[0]))
# p_id_in_file = np.array(p_id_in_file)
# dataset_p_id = p_id_in_file

# labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])

# print('Finished feature extraction from ', len(dataset_p_id), ' files')
# unique_elements, counts_elements = np.unique(labels, return_counts=True)
# print(np.asarray((unique_elements, counts_elements)))

# le = LabelEncoder()
# i_labels = le.fit_transform(labels)
# oh_labels = to_categorical(i_labels)

# trainset_p_id, testset_p_id, train_label, test_label = train_test_split(dataset_p_id, oh_labels, stratify=i_labels,
#                                                     test_size=0.2, random_state = 42)


# def extract_image(file_name):
#     #file_name=
#     im = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
#     #im = plt.resize(im, (640, 480))
#     #print('file_name=',file_name)
#     return im

# # preprocess trainset
# waveform_trainset=[]
# for p_id in trainset_p_id:
#     for file_name in listdir(mypath):
#         if (int(file_name.split('_')[0]) == p_id):

#             extract_wave_data = extract_image((mypath+file_name))
#             waveform_trainset.append(extract_wave_data)

# waveform_trainset = np.array(waveform_trainset)
# waveform_trainset = np.reshape(waveform_trainset, (*waveform_trainset.shape, 1))
# waveform_trainset = np.transpose(waveform_trainset, (0,3,1,2))

# # preprocess testset
# waveform_testset=[]
# for p_id in testset_p_id:
#     for file_name in listdir(mypath):
#         if (int(file_name.split('_')[0]) == p_id):

#             extract_wave_data = extract_image((mypath+file_name))
#             waveform_testset.append(extract_wave_data)
# waveform_testset = np.array(waveform_testset)
# waveform_testset = np.reshape(waveform_testset, (*waveform_testset.shape, 1))
# waveform_testset = np.transpose(waveform_testset, (0,3,1,2))

# np.save('waveform_trainset_5classes.npy',waveform_trainset)
# np.save('waveform_testset_5classes.npy',waveform_testset)
# np.save('train_label_wave.npy',train_label)
# np.save('test_label_wave.npy',test_label)

waveform_trainset = np.load('waveform_trainset_5classes.npy')
waveform_testset = np.load('waveform_testset_5classes.npy')
train_label = np.load('train_label.npy')
test_label = np.load('test_label.npy')

# # construct model on cuda if available
use_cuda = torch.cuda.is_available()

# loss
criterion = nn.MultiLabelSoftMarginLoss()
# wave CNN model
class wave_cnn(nn.Module):

    def __init__(self):

        super(wave_cnn, self).__init__()

        self.wave_conv1 = nn.Conv2d(1, 16, kernel_size=(2,10))  
        self.wave_bn1 = torch.nn.BatchNorm2d(16)
        self.wave_pool1 = nn.MaxPool2d(2)  
        self.wave_conv2 = nn.Conv2d(16, 32, kernel_size=(2,10))  
        self.wave_bn2 = torch.nn.BatchNorm2d(32)
        self.wave_pool2 = nn.MaxPool2d(2) 
        self.wave_conv3 = nn.Conv2d(32, 64, kernel_size=(2,10))  
        self.wave_bn3 = torch.nn.BatchNorm2d(64)
        self.wave_pool3 = nn.MaxPool2d(2)  
        self.wave_conv4 = nn.Conv2d(64, 128, kernel_size=2)  
        self.wave_bn4 = torch.nn.BatchNorm2d(128)
        self.wave_pool4 = nn.MaxPool2d(2)  
        self.wave_conv5 = nn.Conv2d(128, 256, kernel_size=2)  
        self.wave_bn5 = torch.nn.BatchNorm2d(256)
        self.wave_pool5 = nn.MaxPool2d(2)  
        self.wave_conv6 = nn.Conv2d(256, 512, kernel_size=2) 
        self.wave_bn6 = torch.nn.BatchNorm2d(512)
        self.wave_pool6 = nn.MaxPool2d(2)  
        self.wave_global_pool = nn.AdaptiveAvgPool2d(1)
        self.wave_fc1 = nn.Linear(512, 256)
        self.wave_fc2 = nn.Linear(256, 128)
        self.wave_fc_bn1 = torch.nn.BatchNorm1d(128)
        self.wave_fc3 = nn.Linear(128, classes)
        self.wave_relu = nn.ReLU()
       
    
    def forward(self, wave_x):
        
        wave_x = self.wave_relu(self.wave_bn1(self.wave_conv1(wave_x)))
        wave_x = self.wave_pool1(wave_x) 
        wave_x = self.wave_relu(self.wave_bn2(self.wave_conv2(wave_x)))
        wave_x = self.wave_pool2(wave_x) 
        wave_x = self.wave_relu(self.wave_bn3(self.wave_conv3(wave_x)))
        wave_x = self.wave_pool3(wave_x) 
        wave_x = self.wave_relu(self.wave_bn4(self.wave_conv4(wave_x)))
        wave_x = self.wave_pool4(wave_x) 
        wave_x = self.wave_relu(self.wave_bn5(self.wave_conv5(wave_x)))
        wave_x = self.wave_pool5(wave_x) 
        wave_x = self.wave_relu(self.wave_bn6(self.wave_conv6(wave_x)))
        wave_x = self.wave_pool6(wave_x)    
        wave_x = self.wave_global_pool(wave_x)
        wave_x = wave_x.view(wave_x.size(0), -1)
        wave_x = self.wave_fc1(wave_x) 
        wave_x = self.wave_fc2(wave_x)
        wave_x = self.wave_fc_bn1(wave_x) # feature size 128

        wave_feature_x = wave_x # feature size 128
        wave_x = self.wave_fc3(wave_x)

        return wave_x, wave_feature_x


wave_cnn = wave_cnn()
wave_cnn.cuda()
wave_optimizer = optim.Adam(wave_cnn.parameters(), lr=learning_rate)
wave_cnn.load_state_dict(torch.load("/home/iichsk/workspace/weight/Waveform_cnn2_weight_100_128_44.pt"))

# Waveform CNN train
def wave_train(waveform_trainset, label):
    z = np.random.permutation(waveform_trainset.shape[0])
    trn_loss_list = []
    print('Waveform CNN train start!!!!!')
    wave_cnn.train()
    for epoch in range(wave_num_epochs):
        trn_loss = 0.0
        for i in range(int(waveform_trainset.shape[0] / batch_size)):
            wave_input = torch.Tensor(waveform_trainset[z[i*batch_size:(i+1)* batch_size], :, :, :]).cuda()
            label_torch =  torch.Tensor(label[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            # grad init
            wave_optimizer.zero_grad()
            wave_cnn_output, _ = wave_cnn(wave_input)
            # calculate loss
            loss = criterion(wave_cnn_output, label_torch)
            # back propagation
            loss.backward()
            # weight update
            wave_optimizer.step()
            # trn_loss summary
            trn_loss += loss.item()
           # del (memory issue)
            '''del loss
            del model_output'''

            # 학습과정 출력
        if (epoch + 1) % 50 == 0:  #
            print("epoch: {}/{} | trn loss: {:.8f}".format(
                epoch + 1, wave_num_epochs, trn_loss / 100))
            trn_loss_list.append(trn_loss / 100)

    count_t = time.time()
    torch.save(wave_cnn.state_dict(), "/home/iichsk/workspace/weight/Waveform_cnn2_weight_{}_{}_{}.pt".format(wave_num_epochs, batch_size, count_t))
    print("Waveform CNN Model's state_dict saved")

# Waveform CNN test
def wave_cnn_test(waveform_testset, label):
    wave_cnn.eval()
    predictions = np.zeros((int(waveform_testset.shape[0]), classes))
    wave_features = np.zeros((int(waveform_testset.shape[0]), 128))

    for j in range(int(waveform_testset.shape[0])):
        wave_input = torch.Tensor(waveform_testset[j:(j+1), :, :, :]).cuda()
        test_output, wave_feature = wave_cnn(wave_input)
        predictions[j:j+1,:] = test_output.detach().cpu().numpy()
        wave_features[j:j+1,:] = wave_feature.data.cpu().numpy()

    return predictions, wave_features

# Waveform CNN part
# wave_train(waveform_trainset, train_label)
wave_predictions, wave_test_features = wave_cnn_test(waveform_testset, test_label)
wave_test_features_df = pd.DataFrame(wave_test_features)
count_t = time.time()
wave_test_features_df.to_excel("/home/iichsk/workspace/respiratory_classify/result/wave_test_features{}.xlsx".format(count_t), sheet_name = 'sheet1')

# CNN models report
wave_predictions_arg = np.argmax(wave_predictions, axis=1)  
target = np.argmax(test_label, axis=1)  # true classes
c_names = ['both','crackles','d_normal','h_normal','wheezes']
print('#'*10,'Waveform CNN report','#'*10)
print(classification_report(target, wave_predictions_arg, target_names=c_names))
print(confusion_matrix(target, wave_predictions_arg))

# result save to excell
wave_predictions_arg=np.expand_dims(wave_predictions_arg, axis=1)
target=np.expand_dims(target, axis=1)
result_for_excell = np.concatenate((wave_predictions_arg,target),1)
df = pd.DataFrame(result_for_excell,columns=['Waveform', 'target'])
count_t = time.time()
df.to_excel("/home/iichsk/workspace/respiratory_classify/result/waveform_result{}.xlsx".format(count_t), sheet_name = 'sheet1')
