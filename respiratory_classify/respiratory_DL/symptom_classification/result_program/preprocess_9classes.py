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

# dataset load
p_diag = pd.read_csv("/home/iichsk/workspace/respiratory_classify/respiratory_DL/symptom_classification/result_program/symptom_best_result.csv",header=None) # patient diagnosis csv file
p_id_in_file = [] # patient IDs corresponding to each file
cnt2 = 0
for x in p_diag[0]:
    if(int(p_diag.iloc[cnt2,7]) == 0):
        print(cnt2)
        p_id_in_file.append(p_diag.iloc[cnt2,2:])
    cnt2+=1
p_id_in_file = np.array(p_id_in_file)
print(p_id_in_file)
dataset_p_id = p_id_in_file
dataset_p_id=pd.DataFrame(dataset_p_id)
dataset_p_id.to_csv("S_error.csv",index=False)
