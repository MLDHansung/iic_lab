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

matching_result_testset = pd.read_csv("matching_result_testset.csv",header=None) # patient diagnosis csv file
matching_result_testset_np =np.array(matching_result_testset)
predictions = matching_result_testset_np[1:,1]
target = matching_result_testset_np[1:,2]
print(predictions.shape)
print(target.shape)
c_names = ['Brc_B','Brc_C','Brc_N','Brc_W','Brl_B','Brl_C','Brl_N','Brl_W','C_B','C_C','C_N','C_W','H_C','H_N','P_N','P_W','U_C','U_N','U_W']
# Classification Report
print('#'*10,'Triple Ensemble CNN report','#'*10)
print(classification_report(target, predictions, target_names=c_names))
# Confusion Matrix
print(confusion_matrix(target, predictions))

