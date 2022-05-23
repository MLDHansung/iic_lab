from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler


sc = MinMaxScaler()

class lstm_ae_preprocessing:
	#create sequence

	def sliding_windows(self, data, seq_length):
	    x = []
	    y = []

	    for i in range(len(data) - seq_length -1):
	        _x = data[i:(i + seq_length-1)]
	        x.append(_x)

	    return np.array(x)

	
	def load_data(self, path, seq_len):
		dataset = pd.read_csv(path)
		train_test_dataset = dataset.filter(
		    items=['targetJoint1', 'targetJoint2', 'path', 'seqNum', 'joint1', 'joint2','joint1_difference','joint2_difference'])
		self.seq_len=seq_len
		len_dataset=int(len(train_test_dataset))
		train_size=int(len_dataset*0.7)
		test_set = train_test_dataset.iloc[0:, :]
		training_set = train_test_dataset.iloc[0:, :]
		training_set = sc.fit_transform(training_set)
		test_set = sc.fit_transform(test_set)
		train_x = self.sliding_windows(training_set, seq_len + 1)
		test_x = self.sliding_windows(test_set, seq_len + 1)
		trainX = np.array(train_x)  
		trainX = np.swapaxes(trainX, 0, 1)  
		testX = np.array(test_x)

		testX = np.swapaxes(testX, 0, 1)

		

		return trainX, testX

	def normalize_inverse(self, trainX, testX):
		print('trainX',trainX.shape)
		trainX_tmp = np.zeros((int(trainX.shape[1]), (int(trainX.shape[2]))))
		trainX_inverse = np.zeros((self.seq_len, int(trainX.shape[1]),  int(trainX.shape[2])))
		testX_tmp = np.zeros((int(testX.shape[1]),  (int(testX.shape[2]))))
		testX_inverse = np.zeros((self.seq_len, int(testX.shape[1]),  (int(testX.shape[2]))))
		for iii in range(self.seq_len):
		    trainX_tmp[:, :] = trainX[iii, :, :]
		    trainX_inverse[iii, :, :] = sc.inverse_transform(trainX_tmp)
		    testX_tmp[:, :] = testX[iii, :, :]
		    testX_inverse[iii, :, :] = sc.inverse_transform(testX_tmp)

		return trainX_inverse, testX_inverse