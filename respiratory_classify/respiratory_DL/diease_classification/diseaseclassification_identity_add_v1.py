from os import listdir
from os.path import isfile, join
from PIL import Image
import librosa
import librosa.display
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import cv2
import time
# KMP_AFFINITY="none"
# os.environ['KMP_AFFINITY'] = '0'
batch_size = 128
learning_rate = 1e-3
filter_size1 = 10
filter_size2 = 2
classes = 6
spt_dropout = 0.325
img_dropout = 0.325
mypath = "/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files2/"
mypath_waveform = "/home/iichsk/workspace/dataset/full_image/audio_and_txt_files2/"
p_diag = pd.read_csv("/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv",header=None)

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]

p_id_in_file = [] # patient IDs corresponding to each file
file_full_name = []
for name in filenames:
    p_id_in_file.append(int(name[:3]))
    file_full_name.append(name)
p_id_in_file = np.array(p_id_in_file)

filepaths = [join(mypath, f) for f in filenames] # full paths of files

labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file])

dataset_p_id = np.array(file_full_name)
dataset_p_id = np.delete(dataset_p_id, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)
labels = np.delete(labels, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)

print('Finished feature extraction from ', len(dataset_p_id), ' files')
unique_elements, counts_elements = np.unique(labels, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

le = LabelEncoder()
i_labels = le.fit_transform(labels)
oh_labels = to_categorical(i_labels)

trainset_p_id, testset_p_id, train_label, test_label = train_test_split(dataset_p_id, oh_labels, stratify=i_labels,
                                                    test_size=0.2, random_state = 42)

max_pad_len = 862 # to make the length of all MFCC equal
def extract_MFCC(file_name):
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

mfcc_trainset=[]
waveform_trainset=[]
cnt=0
for file_name in trainset_p_id:
    extract_mfcc_data = extract_MFCC((mypath+file_name))
    mfcc_trainset.append(extract_mfcc_data)
    cnt+=1
    for file_name_wave in listdir(mypath_waveform):
        if ((file_name_wave.split('.')[0]) == (file_name.split('.')[0])):
            extract_waveform_data = cv2.imread(mypath_waveform+file_name_wave, cv2.IMREAD_GRAYSCALE)
            waveform_trainset.append(extract_waveform_data)
    if(cnt%100==0):
        print('Data extracting no. {}'.format(cnt))

mfcc_trainset = np.array(mfcc_trainset)
waveform_trainset = np.array(waveform_trainset)

# preprocess testset
mfcc_testset=[]
waveform_testset=[]
for file_name in testset_p_id:
    extract_mfcc_data = extract_MFCC((mypath+file_name))
    mfcc_testset.append(extract_mfcc_data)

    for file_name_wave in listdir(mypath_waveform):
        if ((file_name_wave.split('.')[0]) == (file_name.split('.')[0])):
            extract_waveform_data = cv2.imread(mypath_waveform+file_name_wave, cv2.IMREAD_GRAYSCALE)
            waveform_testset.append(extract_waveform_data)

mfcc_testset = np.array(mfcc_testset)
waveform_testset = np.array(waveform_testset)

mfcc_trainset = np.reshape(mfcc_trainset, (*mfcc_trainset.shape, 1))
mfcc_testset = np.reshape(mfcc_testset, (*mfcc_testset.shape, 1))
waveform_trainset = np.reshape(waveform_trainset, (*waveform_trainset.shape, 1))
waveform_testset = np.reshape(waveform_testset, (*waveform_testset.shape, 1))

mfcc_trainset = np.transpose(mfcc_trainset, (0,3,1,2))
mfcc_testset = np.transpose(mfcc_testset, (0,3,1,2))
waveform_trainset = np.transpose(waveform_trainset, (0,3,1,2))
waveform_testset = np.transpose(waveform_testset, (0,3,1,2))

count_t = time.time()
np.save('/home/iichsk/workspace/respiratory_classify/respiratory_DL/diease_classification/numpy_dataset/trainset/mfcc_trainset_{}.npy'.format(count_t),mfcc_trainset)
np.save('/home/iichsk/workspace/respiratory_classify/respiratory_DL/diease_classification/numpy_dataset/testset/mfcc_testset_{}.npy'.format(count_t),mfcc_testset)

np.save('/home/iichsk/workspace/respiratory_classify/respiratory_DL/diease_classification/numpy_dataset/trainset/waveform_trainset_{}.npy'.format(count_t),waveform_trainset)
np.save('/home/iichsk/workspace/respiratory_classify/respiratory_DL/diease_classification/numpy_dataset/testset/waveform_testset_{}.npy'.format(count_t),waveform_testset)

np.save('/home/iichsk/workspace/respiratory_classify/respiratory_DL/diease_classification/numpy_dataset/trainset/train_label_{}.npy'.format(count_t),train_label)
np.save('/home/iichsk/workspace/respiratory_classify/respiratory_DL/diease_classification/numpy_dataset/testset/test_label_{}.npy'.format(count_t),test_label)

print('Finished MFCC feature extraction from ', len(mfcc_trainset)+len(mfcc_testset), ' files')
print('Finished waveform feature extraction from ', len(waveform_trainset)+len(waveform_testset), ' files')

# construct model on cuda if available
use_cuda = torch.cuda.is_available()

# loss
criterion = nn.MultiLabelSoftMarginLoss()
class ensemble_cnn(nn.Module):

    def __init__(self):
        # 항상 torch.nn.Module을 상속받고 시작
        super(ensemble_cnn, self).__init__()

        self.spt_conv1 = nn.Conv2d(1, 16, kernel_size=4)  # 16@39*861
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
        self.spt_relu = nn.ReLU()

        self.img_conv1 = nn.Conv2d(1, 16, kernel_size=10)  # 16@39*861
        self.img_pool1 = nn.MaxPool2d(2)  # 16@19*430
        self.img_dropout1 = nn.Dropout(img_dropout)
        self.img_conv2 = nn.Conv2d(16, 32, kernel_size=2)  # 32@18*429
        self.img_pool2 = nn.MaxPool2d(2)  # 32@9*214
        self.img_dropout2 = nn.Dropout(img_dropout)
        self.img_conv3 = nn.Conv2d(32, 64, kernel_size=2)  # 64@8*213
        self.img_pool3 = nn.MaxPool2d(2)  # 64@4*106
        self.img_dropout3 = nn.Dropout(img_dropout)
        self.img_conv4 = nn.Conv2d(64, 128, kernel_size=2)  # 128@3*105
        self.img_pool4 = nn.MaxPool2d(2)  # 128@1*52
        self.img_dropout4 = nn.Dropout(img_dropout)
        self.img_global_pool = nn.AdaptiveAvgPool2d(1)
        self.img_relu = nn.ReLU()

        self.ensemble_fc1 = nn.Linear(128, 64)
        self.ensemble_bn1 = torch.nn.BatchNorm1d(64)
        self.ensemble_fc2 = nn.Linear(64, classes)
        self.ensemble_relu = nn.ReLU()
        self.ensemble_dropout1 = nn.Dropout(0.4)
    
    def forward(self, spt_x, img_x):

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
        spt_x = spt_x.view(spt_x.size(0), -1)

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
        img_x = self.img_global_pool(img_x)
        img_x = img_x.view(img_x.size(0), -1)

        ensemble_x = spt_x + img_x
        ensemble_x = self.ensemble_fc1(ensemble_x)  #
        ensemble_x = self.ensemble_bn1(ensemble_x)
        ensemble_x = self.ensemble_dropout1(ensemble_x)
        ensemble_x = self.ensemble_fc2(ensemble_x)  #

        return ensemble_x


ensemble_cnn = ensemble_cnn()
ensemble_cnn.cuda()
# backpropagation method
# ensemble_cnn.load_state_dict(torch.load("/home/iichsk/workspace/respiratory_classify/weight/ensemble_cnn/ensemble_cnn_weight_"))
ensemble_cnn_optimizer = optim.Adam(ensemble_cnn.parameters(), lr=learning_rate)

def ensemble_cnn_train(mfcc_trainset,wave_trainset,train_label):
    num_epochs = 450
    z = np.random.permutation(mfcc_trainset.shape[0])

    trn_loss_list = []
    print('Ensemble CNN train start!!!!!')
    ensemble_cnn.train()
    for epoch in range(num_epochs):
        trn_loss = 0.0
        for i in range(int(mfcc_trainset.shape[0] / batch_size)):
            mfcc_input = torch.Tensor(mfcc_trainset[z[i*batch_size:(i+1)* batch_size], :, :, :]).cuda()
            wave_input = torch.Tensor(wave_trainset[z[i*batch_size:(i+1)* batch_size], :, :, :]).cuda()
            label =  torch.Tensor(train_label[z[i*batch_size:(i+1)* batch_size], :]).cuda()
            # grad init
            ensemble_cnn_optimizer.zero_grad()
            model_output = ensemble_cnn(mfcc_input,wave_input)
            
            # calculate loss

            ensemble_cnn_loss = criterion(model_output, label)
            # back propagation
            ensemble_cnn_loss.backward(retain_graph=True)

            # weight update
            ensemble_cnn_optimizer.step()

            # trn_loss summary
            trn_loss += ensemble_cnn_loss.item()
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
    torch.save(ensemble_cnn.state_dict(), "/home/iichsk/workspace/weight/ensemble_cnn/ensemble_cnn_weight_{}_{}_{}.pt".format(num_epochs, batch_size, count_t))

def ensemble_cnn_test(mfcc_testset,wave_testset):
       
    ensemble_cnn.eval()
    test_loss = 0.0
    predictions = np.zeros((int(mfcc_testset.shape[0]), classes))

    for j in range(int(mfcc_testset.shape[0])):
        mfcc_input = torch.Tensor(mfcc_testset[j:(j+1), :, :, :]).cuda()
        wave_input = torch.Tensor(wave_testset[j:(j+1), :, :, :]).cuda()

        test_output = ensemble_cnn(mfcc_input, wave_input)

        predictions[j:j+1,:] = test_output.detach().cpu().numpy()

    return predictions


ensemble_cnn_train(mfcc_trainset,waveform_trainset,train_label)
predictions = ensemble_cnn_test(mfcc_testset,waveform_testset)
classpreds = np.argmax(predictions, axis=1)  # predicted classes
target = np.argmax(label_test, axis=1)  # true classes
#c_names = ['Abnormal', 'Healthy']
c_names = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Pneumonia', 'URTI', 'Healthy']

# Classification Report
print(classification_report(target, classpreds, target_names=c_names))
