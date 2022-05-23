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
classes = 4
MFCC_dropout = 0.3
scalo_dropout = 0.3
wave_dropout = 0.3
mfcc_num_epochs = 2000
wave_num_epochs = 1000
scalo_num_epochs = 1000
ensem_num_epochs = 300

#### dataset load
p_diag = pd.read_csv("/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/csv_data/data2.csv",header=None) # patient diagnosis csv file
p_diag_np=p_diag.to_numpy()

filename = "init"
excell_list = []
p_list= []
for i in range(p_diag_np.shape[0]):
    if(filename != p_diag_np[i,8]):
        excell_list.append(p_list)
        p_list =[]
        filename = p_diag_np[i,8]
        p_list.append(p_diag_np[i,8])
        p_list.append(p_diag_np[i,9])
        p_list.append(p_diag_np[i,1])
    else:
        p_list.append(p_diag_np[i,1])
del excell_list[0]
df = pd.DataFrame.from_records(excell_list)
df.to_excel('/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/result/disease_symptom_timeline.xlsx')
# print('Finished feature extraction from ', len(dataset_p_id), ' files')
# unique_elements, counts_elements = np.unique(labels, return_counts=True)
# print(np.asarray((unique_elements, counts_elements)))
