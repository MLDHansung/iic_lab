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

        for i in range(len(data) - seq_length+1):
            _x = data[i:(i + seq_length)]
            _y = data[i]
            x.append(_x)
            y.append(_y)

        return np.array(x), np.array(y)


    def sliding_windows_seq(self, data, seq_length):
        x = []
        y = []

        for i in range(len(data) - seq_length+1):
            if((i)%seq_length==0):
                #print('data=',data[i:(i + seq_length),6])
                _x = data[i:(i + seq_length),:]
                _y = data[i]
                x.append(_x)
                y.append(_y)
                #print('x',x)
        return np.array(x), np.array(y)
    


    def load_data(self, data_path, seq_len):
        dataset = pd.read_csv(data_path)
        train_test_dataset = dataset.filter(
            items=['targetJ1', 'targetJ2', 'targetJ3', 'targetJ4', 'targetJ5', 'targetJ6','path','seqNum',
            'joint1','joint2','joint3','joint4','joint5','joint6',
            'jointVel1','jointVel2','jointVel3','jointVel4','jointVel5','jointVel6',
            'diffJoint1','diffJoint2','diffJoint3','diffJoint4','diffJoint5','diffJoint6'])   

        label_set = dataset.filter(items=['class'])
        data_label_set = label_set[:]
        data_set = train_test_dataset.iloc[:, :]    
        data_set = np.array(data_set)  
        #print('data_set',data_set.shape)

        data_label_Y = np.array(data_label_set)  
        data_x, _ = self.sliding_windows(data_set, seq_len)
        _, data_label_Y = self.sliding_windows(data_label_Y, seq_len)
        dataX = np.array(data_x) 

        return dataX, data_label_Y

    def load_data2(self, data_path, seq_len):
        dataset = pd.read_csv(data_path)
        train_test_dataset = dataset.filter(
            items=['targetJ1', 'targetJ2', 'targetJ3', 'targetJ4', 'targetJ5', 'targetJ6','path','seqNum',
            'joint1','joint2','joint3','joint4','joint5','joint6',
            'jointVel1','jointVel2','jointVel3','jointVel4','jointVel5','jointVel6',
            'diffJoint1','diffJoint2','diffJoint3','diffJoint4','diffJoint5','diffJoint6','class'])       
        data_seqNum = dataset.filter(items=['seqNum'])
        data_seqNum = np.array(data_seqNum)
        label_set = train_test_dataset.filter(items=['class'])
        del train_test_dataset['class']
        data_label_set = label_set[:]
        data_set = train_test_dataset.iloc[:, :]
        data_set = np.array(data_set)  
        data_set_temp = []
        data_label_set = np.array(data_label_set)  
        data_label_set_temp = []
        iii = 0
        for i in range(0,len(dataset)-1):
            if (data_seqNum[i]==1):
                for ii in range(seq_len):
                    a=i
                    data_set_temp.append(data_set[a+ii:a+ii+1,:])
                    data_label_set_temp.append(data_label_set[a+ii:a+ii+1])
           
        data_set = data_set_temp
        data_set_np = np.array(data_set)
        data_set_np = np.squeeze(data_set_np, axis=1)

        #df=pd.DataFrame.from_records(data_set_np[:,:])
        #df.to_excel('{}_data_set.xlsx'.format(data_path))

        data_label_set = data_label_set_temp
        data_label_set_np = np.array(data_label_set)
        
        data_x, _ = self.sliding_windows_seq(data_set_np, seq_len)
        _, data_label_Y = self.sliding_windows_seq(data_label_set_np, seq_len)
        #data_x = np.squeeze(data_x, axis=2)
        data_label_Y = np.squeeze(data_label_Y, axis=2)
        #print('data_x',data_x[0,:,6])
        print(data_x.shape)

        #df2=pd.DataFrame.from_records(data_x[0:,:,6])
        #df2.to_excel('{}_sliding.xlsx'.format(data_path))
        return data_x, data_label_Y

    def load_data3(self, data_path, seq_len):
        dataset = pd.read_csv(data_path)
        train_test_dataset = dataset.filter(
            items=['targetJ1', 'targetJ2', 'targetJ3', 'targetJ4', 'targetJ5', 'targetJ6','path','seqNum',
            'joint1','joint2','joint3','joint4','joint5','joint6',
            'jointVel1','jointVel2','jointVel3','jointVel4','jointVel5','jointVel6',
            'diffJoint1','diffJoint2','diffJoint3','diffJoint4','diffJoint5','diffJoint6','class'])       
        path_num = train_test_dataset['path'] == 5
        train_test_dataset = train_test_dataset[path_num]
        label_set = train_test_dataset.filter(items=['class'])
        del train_test_dataset['class']
        data_set = train_test_dataset.iloc[:, :]
        data_set = np.array(data_set)
        data_label_set = label_set[:]
        data_set_temp = np.empty((1,data_set.shape[1]))
        data_label_set = np.array(data_label_set)  
        data_label_set_temp = np.empty((1,1))
        data_seqNum = data_set[:,7:8]
        data_seqNum = np.array(data_seqNum)

        data_seqNum_frame=pd.DataFrame.from_records(data_seqNum[:,:])
        data_seqNum_frame.to_excel('{}_data_seqNum.xlsx'.format(data_path))
        for i in range(0,len(data_set)-1-seq_len):
            if (data_seqNum[i]==1):
                for ii in range(seq_len):
                    data_set_temp=np.vstack((data_set_temp,data_set[i+ii:i+ii+1,:]))
                    data_label_set_temp=np.vstack((data_label_set_temp,data_label_set[i+ii:i+ii+1]))
                
 
        
        data_set_np = np.delete(data_set_temp,0,axis=0)
        data_label_set_np = np.delete(data_label_set_temp,0,axis=0)
        data_set_temp_frame=pd.DataFrame.from_records(data_set_np[:,:])
        data_set_temp_frame.to_excel('{}_data_set_temp.xlsx'.format(data_path))

        data_x, _ = self.sliding_windows_seq(data_set_np, seq_len)
        _, data_label_Y = self.sliding_windows_seq(data_label_set_np, seq_len)

        df2=pd.DataFrame.from_records(data_x[:,0,:])
        df2.to_excel('{}_sliding.xlsx'.format(data_path))
        return data_x, data_label_Y

    def load_data4(self, data_path, seq_len):
        dataset = pd.read_csv(data_path)
        train_test_dataset = dataset.filter(
            items=['targetJ1', 'targetJ2', 'targetJ3', 'targetJ4', 'targetJ5', 'targetJ6','path','seqNum',
            'joint1','joint2','joint3','joint4','joint5','joint6',
            'jointVel1','jointVel2','jointVel3','jointVel4','jointVel5','jointVel6',
            'diffJoint1','diffJoint2','diffJoint3','diffJoint4','diffJoint5','diffJoint6'])   

        label_set = dataset.filter(items=['class'])
        data_label_set = label_set[:]
        data_set = train_test_dataset.iloc[:, :]    
        data_set = np.array(data_set)  
        #print('data_set',data_set.shape)

        data_label_Y = np.array(data_label_set)  
        data_x, _ = self.sliding_windows_seq(data_set, seq_len)
        _, data_label_Y = self.sliding_windows_seq(data_label_Y, seq_len)
        dataX = np.array(data_x) 

        return dataX, data_label_Y
    def dataset_split(self, dataset, data_label):
        dataset_temp=np.zeros((int(dataset.shape[0]),int(dataset.shape[1]),int(dataset.shape[2])))
        for i in range(dataset.shape[1]):
            dataset_temp[:,i,:] = sc.fit_transform(dataset[:,i,:])
        i_labels = le.fit_transform(data_label)
        data_label = to_categorical(i_labels)
        trainX, testX, train_label_Y, test_label_Y = train_test_split(dataset, data_label, stratify=i_labels,
                                                            test_size=0.2, random_state = 42)
        trainX = np.swapaxes(trainX, 0, 1)  
        testX = np.swapaxes(testX, 0, 1)  

        return trainX, testX, train_label_Y, test_label_Y

    def load_data5(self, data_path, seq_len):
        dataset = pd.read_csv(data_path)
        train_test_dataset = dataset.filter(
            items=['targetJ1', 'targetJ2', 'targetJ3', 'targetJ4', 'targetJ5', 'targetJ6','path','seqNum',
            'joint1','joint2','joint3','joint4','joint5','joint6',
            'jointVel1','jointVel2','jointVel3','jointVel4','jointVel5','jointVel6',
            'diffJoint1','diffJoint2','diffJoint3','diffJoint4','diffJoint5','diffJoint6','class'])       
        data_seqNum = dataset.filter(items=['seqNum'])
        data_seqNum = np.array(data_seqNum)
        label_set = train_test_dataset.filter(items=['class'])
        del train_test_dataset['class']
        data_label_set = label_set[:]
        data_set = train_test_dataset.iloc[:, :]
        data_set = np.array(data_set)  
        data_set_temp = []
        data_label_set = np.array(data_label_set)  
        data_label_set_temp = []
        iii = 0
        for i in range(0,len(dataset)-1):
            if (data_seqNum[i]==1):
                for ii in range(seq_len):
                    a=i
                    data_set_temp.append(data_set[a+ii:a+ii+1,:])
                    data_label_set_temp.append(data_label_set[a+ii:a+ii+1])
           
        data_set = data_set_temp
        data_set_np = np.array(data_set)
        data_set_np = np.squeeze(data_set_np, axis=1)

        #df=pd.DataFrame.from_records(data_set_np[:,:])
        #df.to_excel('{}_data_set.xlsx'.format(data_path))

        data_label_set = data_label_set_temp
        data_label_set_np = np.array(data_label_set)
        data_set_np = sc.fit_transform(data_set_np)
        data_x, _ = self.sliding_windows_seq(data_set_np, seq_len)
        _, data_label_Y = self.sliding_windows_seq(data_label_set_np, seq_len)
        #data_x = np.squeeze(data_x, axis=2)
        data_label_Y = np.squeeze(data_label_Y, axis=2)
        #print('data_x',data_x[0,:,6])
        print(data_x.shape)

        #df2=pd.DataFrame.from_records(data_x[0:,:,6])
        #df2.to_excel('{}_sliding.xlsx'.format(data_path))
        return data_x, data_label_Y
    def load_data6(self, data_path, seq_len):
        dataset = pd.read_csv(data_path)
        train_test_dataset = dataset.filter(
            items=['targetJ1', 'targetJ2', 'targetJ3', 'targetJ4', 'targetJ5', 'targetJ6','path','seqNum',
            'joint1','joint2','joint3','joint4','joint5','joint6',
            'jointVel1','jointVel2','jointVel3','jointVel4','jointVel5','jointVel6',
            'diffJoint1','diffJoint2','diffJoint3','diffJoint4','diffJoint5','diffJoint6','class'])       
        data_seqNum = dataset.filter(items=['seqNum'])
        data_seqNum = np.array(data_seqNum)
        label_set = train_test_dataset.filter(items=['class'])
        del train_test_dataset['class']
        data_label_set = label_set[:]
        data_set = train_test_dataset.iloc[:, :]
        data_set = np.array(data_set)  
        print('data_set',data_set.shape)

        data_set_temp = []
        data_label_set = np.array(data_label_set)  
        data_label_set_temp = []
        for i in range(0,len(dataset)-1):
            if (data_seqNum[i]==1):
                for ii in range(seq_len):
                    a=i
                    data_set_temp.append(data_set[a+ii:a+ii+1,:])
                    data_label_set_temp.append(data_label_set[a+ii:a+ii+1])
           
        data_set = data_set_temp
        data_set_np = np.array(data_set)
        data_set_np = np.squeeze(data_set_np, axis=1)

        #df=pd.DataFrame.from_records(data_set_np[:,:])
        #df.to_excel('{}_data_set.xlsx'.format(data_path))
        print(data_set_np.shape)

        data_label_set = data_label_set_temp
        data_label_set_np = np.array(data_label_set)
        data_set_np = sc.fit_transform(data_set_np)
        data_x, _ = self.sliding_windows(data_set_np, seq_len)
        _, data_label_Y = self.sliding_windows(data_label_set_np, seq_len)
        #data_x = np.squeeze(data_x, axis=2)
        data_label_Y = np.squeeze(data_label_Y, axis=2)
        #print('data_x',data_x[0,:,6])
        print(data_x.shape)

        #df2=pd.DataFrame.from_records(data_x[0:,:,6])
        #df2.to_excel('{}_sliding.xlsx'.format(data_path))
        return data_x, data_label_Y

    def load_data7(self, data_path, seq_len):
        dataset = pd.read_csv(data_path)
        train_test_dataset = dataset.filter(
            items=['targetJ1', 'targetJ2', 'targetJ3', 'targetJ4', 'targetJ5', 'targetJ6','path','seqNum',
            'joint1','joint2','joint3','joint4','joint5','joint6',
            'jointVel1','jointVel2','jointVel3','jointVel4','jointVel5','jointVel6',
            'diffJoint1','diffJoint2','diffJoint3','diffJoint4','diffJoint5','diffJoint6','class'])       
        
        path_num = train_test_dataset['path'] ==3
        train_test_dataset = train_test_dataset[path_num]
        data_seqNum = train_test_dataset.filter(items=['seqNum'])
        data_seqNum = np.array(data_seqNum)
        label_set = train_test_dataset.filter(items=['class'])
        del train_test_dataset['class']
        data_label_set = label_set[:]
        data_set = sc.fit_transform(train_test_dataset)

        data_label_set = label_set[:]
        #data_set = train_test_dataset.iloc[:, :]
        #data_set = np.array(data_set)  
        data_set_temp = []
        data_label_set = np.array(data_label_set)  
        data_label_set_temp = []
        for i in range(0,len(train_test_dataset)-1):
            if (data_seqNum[i]==1):
                for ii in range(seq_len):
                    a=i
                    data_set_temp.append(data_set[a+ii:a+ii+1,:])
                    data_label_set_temp.append(data_label_set[a+ii:a+ii+1])
                    #print(data_set[a+ii:a+ii+1,:])
        data_set = data_set_temp
        data_set_np = np.array(data_set)
        data_set_np = np.squeeze(data_set_np, axis=1)

        #df=pd.DataFrame.from_records(data_set_np[:,:])
        #df.to_excel('{}_data_set.xlsx'.format(data_path))

        data_label_set = data_label_set_temp
        data_label_set_np = np.array(data_label_set)
        data_set_np = sc.fit_transform(data_set_np)
        data_x, _ = self.sliding_windows_seq(data_set_np, seq_len)
        _, data_label_Y = self.sliding_windows_seq(data_label_set_np, seq_len)
        #data_x = np.squeeze(data_x, axis=2)
        data_label_Y = np.squeeze(data_label_Y, axis=2)
        #print('data_x',data_x[0,:,6])
        print(data_x.shape)
        #df2=pd.DataFrame.from_records(data_x[0:,:,6])
        #df2.to_excel('{}_sliding.xlsx'.format(data_path))
        return data_x, data_label_Y

    def load_data8(self, data_path, seq_len):
        dataset = pd.read_csv(data_path)
        train_test_dataset = dataset.filter(
            items=['targetJ1', 'targetJ2', 'targetJ3', 'targetJ4', 'targetJ5', 'targetJ6','path','seqNum',
            'joint1','joint2','joint3','joint4','joint5','joint6',
            'jointVel1','jointVel2','jointVel3','jointVel4','jointVel5','jointVel6',
            'diffJoint','class'])       
        data_seqNum = dataset.filter(items=['seqNum'])
        data_seqNum = np.array(data_seqNum)
        label_set = train_test_dataset.filter(items=['class'])
        del train_test_dataset['class']
        data_set = sc.fit_transform(train_test_dataset)

        data_label_set = label_set[:]
        #data_set = train_test_dataset.iloc[:, :]
        #data_set = np.array(data_set)  
        data_set_temp = []
        data_label_set = np.array(data_label_set)  
        data_label_set_temp = []
        iii = 0
        for i in range(0,len(dataset)-1):
            if (data_seqNum[i]==1):
                for ii in range(seq_len):
                    a=i
                    data_set_temp.append(data_set[a+ii:a+ii+1,:])
                    data_label_set_temp.append(data_label_set[a+ii:a+ii+1])
           
        data_set = data_set_temp
        data_set_np = np.array(data_set)
        data_set_np = np.squeeze(data_set_np, axis=1)

        

        data_label_set = data_label_set_temp
        data_label_set_np = np.array(data_label_set)
        #data_set_np = sc.fit_transform(data_set_np)
        data_x, _ = self.sliding_windows_seq(data_set_np, seq_len)
        _, data_label_Y = self.sliding_windows_seq(data_label_set_np, seq_len)
        #data_x = np.squeeze(data_x, axis=2)
        data_label_Y = np.squeeze(data_label_Y, axis=2)
        #print('data_x',data_x[0,:,6])
        print(data_x.shape)

        #df2=pd.DataFrame.from_records(data_x[0:,:,6])
        #df2.to_excel('{}_sliding.xlsx'.format(data_path))
        return data_x, data_label_Y

    def load_data9(self, data_path, seq_len):
        dataset = pd.read_csv(data_path)
        train_test_dataset = dataset.filter(
            items=['targetJ1', 'targetJ2', 'targetJ3', 'targetJ4', 'targetJ5', 'targetJ6','path','seqNum',
            'joint1','joint2','joint3','joint4','joint5','joint6',
            'jointVel1','jointVel2','jointVel3','jointVel4','jointVel5','jointVel6',
            'diffJoint1','diffJoint2','diffJoint3','diffJoint4','diffJoint5','diffJoint6','class'])         
        data_seqNum = dataset.filter(items=['seqNum'])
        data_seqNum = np.array(data_seqNum)
        label_set = train_test_dataset.filter(items=['class'])
        del train_test_dataset['class']
        data_set = sc.fit_transform(train_test_dataset)

        data_label_set = label_set[:]
        #data_set = train_test_dataset.iloc[:, :]
        #data_set = np.array(data_set)  
        data_set_temp = []
        data_label_set = np.array(data_label_set)  
        data_label_set_temp = []
        iii = 0
        for i in range(0,len(dataset)-1):
            if (data_seqNum[i]==1):
                for ii in range(seq_len):
                    a=i
                    data_set_temp.append(data_set[a+ii:a+ii+1,:])
                    data_label_set_temp.append(data_label_set[a+ii:a+ii+1])
           
        data_set = data_set_temp
        data_set_np = np.array(data_set)
        data_set_np = np.squeeze(data_set_np, axis=1)
        #print('data_set_np',data_set_np.shape)
        df=pd.DataFrame.from_records(data_set_np[:,:])
        df.columns=['targetJ1', 'targetJ2', 'targetJ3', 'targetJ4', 'targetJ5', 'targetJ6','path','seqNum',
            'joint1','joint2','joint3','joint4','joint5','joint6',
            'jointVel1','jointVel2','jointVel3','jointVel4','jointVel5','jointVel6',
            'diffJoint1','diffJoint2','diffJoint3','diffJoint4','diffJoint5','diffJoint6']
        df.to_excel('{}_data_set_seq.xlsx'.format(data_path))

        data_label_set = data_label_set_temp
        data_label_set_np = np.array(data_label_set)
        #data_set_np = sc.fit_transform(data_set_np)
        data_x, _ = self.sliding_windows_seq(data_set_np, seq_len)
        _, data_label_Y = self.sliding_windows_seq(data_label_set_np, seq_len)
        #data_x = np.squeeze(data_x, axis=2)
        data_label_Y = np.squeeze(data_label_Y, axis=2)
        #print('data_x',data_x[0,:,6])

        #df2=pd.DataFrame.from_records(data_x[0:,:,6])
        #df2.to_excel('{}_sliding.xlsx'.format(data_path))
        return data_x, data_label_Y

    def load_data10(self, data_path, seq_len):
        dataset = pd.read_csv(data_path)
        train_test_dataset = dataset.filter(
            items=['path','seqNum',
            'joint1','joint2','joint3','joint4','joint5','joint6',
            'jointVel1','jointVel2','jointVel3','jointVel4','jointVel5','jointVel6',
            'diffJoint1','diffJoint2','diffJoint3','diffJoint4','diffJoint5','diffJoint6','class'])         
        data_seqNum = dataset.filter(items=['seqNum'])
        data_seqNum = np.array(data_seqNum)
        label_set = train_test_dataset.filter(items=['class'])
        del train_test_dataset['class']
        data_set = sc.fit_transform(train_test_dataset)

        data_label_set = label_set[:]
        #data_set = train_test_dataset.iloc[:, :]
        #data_set = np.array(data_set)  
        data_set_temp = []
        data_label_set = np.array(data_label_set)  
        data_label_set_temp = []
        iii = 0
        for i in range(0,len(dataset)-1):
            if (data_seqNum[i]==1):
                for ii in range(seq_len):
                    a=i
                    data_set_temp.append(data_set[a+ii:a+ii+1,:])
                    data_label_set_temp.append(data_label_set[a+ii:a+ii+1])
           
        data_set = data_set_temp
        data_set_np = np.array(data_set)
        data_set_np = np.squeeze(data_set_np, axis=1)
        print('data_set_np',data_set_np.shape)
        df=pd.DataFrame.from_records(data_set_np[:,:])
        df.columns=['path','seqNum',
            'joint1','joint2','joint3','joint4','joint5','joint6',
            'jointVel1','jointVel2','jointVel3','jointVel4','jointVel5','jointVel6',
            'diffJoint1','diffJoint2','diffJoint3','diffJoint4','diffJoint5','diffJoint6']
        df.to_excel('{}_data_set_seq.xlsx'.format(data_path))

        data_label_set = data_label_set_temp
        data_label_set_np = np.array(data_label_set)
        #data_set_np = sc.fit_transform(data_set_np)
        data_x, _ = self.sliding_windows_seq(data_set_np, seq_len)
        _, data_label_Y = self.sliding_windows_seq(data_label_set_np, seq_len)
        #data_x = np.squeeze(data_x, axis=2)
        data_label_Y = np.squeeze(data_label_Y, axis=2)
        #print('data_x',data_x[0,:,6])

        #df2=pd.DataFrame.from_records(data_x[0:,:,6])
        #df2.to_excel('{}_sliding.xlsx'.format(data_path))
        return data_x, data_label_Y
    def dataset_split2(self, dataset, data_label):
        dataset_temp=np.zeros((int(dataset.shape[0]),int(dataset.shape[1]),int(dataset.shape[2])))
        i_labels = le.fit_transform(data_label)
        data_label = to_categorical(i_labels)
        trainX, testX, train_label_Y, test_label_Y = train_test_split(dataset, data_label, stratify=i_labels,
                                                            test_size=0.2, random_state = 42)
        trainX = np.swapaxes(trainX, 0, 1)  
        testX = np.swapaxes(testX, 0, 1)  

        return trainX, testX, train_label_Y, test_label_Y