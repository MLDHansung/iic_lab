from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split

sc = MaxAbsScaler(1)
le = LabelEncoder()

class lstm_preprocessing:

	def sliding_windows(self, data, seq_length):
	    x = []
	    y = []

	    for i in range(len(data) - seq_length -1):
	        if((i)%25==0):
	            _x = data[i:(i + seq_length-1)]
	            _y = data[i]
	            x.append(_x)
	            y.append(_y)

	    return np.array(x), np.array(y)

	def sliding_windows2(self, data, seq_length):
	    x = []
	    y = []

	    for i in range(len(data) - seq_length -1):
	        if((i)%300==0):
	            _x = data[i:(i + seq_length-1)]
	            _y = data[i]
	            x.append(_x)
	            y.append(_y)
	    return np.array(x), np.array(y)


	def load_data(self, data_path, seq_len):
		dataset = pd.read_csv(data_path)
		train_test_dataset = dataset.filter(
		    items=['targetJoint1', 'targetJoint2', 'path', 'seqNum', 'joint1', 'joint2','joint1_difference','joint2_difference'])
		label_set = dataset.filter(items=['class'])
		data_label_set = label_set[:]
		data_set = train_test_dataset.iloc[:, :]
		data_set = sc.fit_transform(data_set)
		data_label_Y = np.array(data_label_set)  
		data_x, _ = self.sliding_windows(data_set, seq_len + 1)
		_, data_label_Y = self.sliding_windows(data_label_Y, seq_len + 1)
		dataX = np.array(data_x) 

		return dataX, data_label_Y

	def load_data2(self, data_path, seq_len):
		dataset = pd.read_csv(data_path)
		train_test_dataset = dataset.filter(
		    items=['targetJoint1', 'targetJoint2', 'path', 'seqNum', 'joint1', 'joint2','joint1_difference','joint2_difference'])
		label_set = dataset.filter(items=['class'])
		data_label_set = label_set[:]
		data_set = train_test_dataset.iloc[:, :]
		data_set = sc.fit_transform(data_set)
		data_set_temp = np.zeros(((1450,8)))
		data_label_set = np.array(data_label_set)  
		data_label_set_temp = np.zeros(((1450,1)))
		iii = 0
		for i in range(0,17400):
		    if i%300==0:
		        for ii in range(25):
		            a=i
		            data_set_temp[iii+ii:iii+ii+1,:]= (data_set[a+ii:a+ii+1,:])
		            data_label_set_temp[iii+ii:iii+ii+1] = (data_label_set[a+ii:a+ii+1])

		        ii+=1
		        iii += ii
		data_set = data_set_temp
		data_label_set = data_label_set_temp
		data_x, _ = self.sliding_windows(data_set, seq_len + 1)
		_, data_label_Y = self.sliding_windows(data_label_set, seq_len + 1)
		dataX = np.array(data_x)  
		data_label_Y = np.array(data_label_Y)  

		return dataX, data_label_Y

	def dataset_split(self, dataset, data_label):
		i_labels = le.fit_transform(data_label)
		data_label = to_categorical(i_labels)
		trainX, testX, train_label_Y, test_label_Y = train_test_split(dataset, data_label, stratify=i_labels,
		                                                    test_size=0.2, random_state = 42)
		trainX = np.swapaxes(trainX, 0, 1)  
		testX = np.swapaxes(testX, 0, 1)  

		return trainX, testX, train_label_Y, test_label_Y
