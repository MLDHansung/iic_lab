import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
# from torch.autograd import Variable
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from vrep_preprocessing import lstm_preprocessing
from lstm_net import Sequence
from keras.utils import to_categorical
import time
import math
from scipy import io


learning_rate = 0.01
input_dim = 20
output_dim = 6
hidden_size = 16
numlayers = 1
dropout_prob_in = 0.3
dropout_prob = dropout_prob_in
dropout_prob_fc = dropout_prob_in
seq_len = 10
batchnorm_var = 2e-2
num_epochs = 500
bsz = 32

def weights_init(m):
    """pytorch의 network를 초기화 하기위한 함수f

    :param Object m: 초기화를 하기 위한 network instance

    :return: None
    """
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(0.0, batchnorm_var)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.normal_(0.0, batchnorm_var)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d):
        size = m.weight.size()
        fan_out = size[0]  # number of rows
        fan_in = size[1]  # number of columns
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)
    elif isinstance(m, nn.ConvTranspose2d):
        size = m.weight.size()
        fan_out = size[0]  # number of rows
        fan_in = size[1]  # number of columns
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)
    elif isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0]  # number of rows
        fan_in = size[1]  # number of columns
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)

    elif isinstance(m, nn.LSTMCell):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)


def init_hidden(bsz, numlayers, hidden_size):
    """LSTM h와 c값을 초기화 하는 함수

    :param int bsz: 초기화할 batch의 크기
    :param int numlayers: LSTM의 layer 수
    :param int hidden_size: LSTM의 hidden 크기

    :return: 초기화된 h_t_enc, c_t_enc, h_t_dec, c_t_dec
    """
    h_t_enc = []
    c_t_enc = []

    for ii in range(numlayers):
        h_t_enc.append(torch.zeros(bsz, hidden_size, requires_grad=False).cuda())
        c_t_enc.append(torch.zeros(bsz, hidden_size, requires_grad=False).cuda())
    return h_t_enc, c_t_enc

################ model initiailize ####################
lp=lstm_preprocessing()
seq = Sequence(input_dim=input_dim, output_dim=output_dim, hidden_size=hidden_size, numlayers=numlayers, dropout_prob_in=dropout_prob_in, dropout_prob=dropout_prob, dropout_prob_fc=dropout_prob_fc)
seq.cuda()
seq.apply(weights_init)

criterion = nn.MultiLabelSoftMarginLoss()
optimizer = torch.optim.Adam(seq.parameters(), lr=learning_rate)

# Train the model
def train(train_data,train_label):
    seq.train()
    for epoch in range(num_epochs+1):
        learning_rate = 0.003 * pow(0.2, math.floor(float(epoch) / 10.0))
        z = np.random.permutation(train_data.shape[1])
       
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        for i in range(int(train_data.shape[1] / bsz)):
            optimizer.zero_grad()
            h_t_enc, c_t_enc = init_hidden(bsz, numlayers, hidden_size)  # hidden state와 cell state 초기화
            input_x = torch.Tensor(train_data[:, z[i*bsz:(i+1)* bsz], :]).cuda()  # seq_len x bsz x input_dim
            label = torch.Tensor(train_label[z[i*bsz:(i+1)* bsz], :]).cuda()
            output = seq(input_x, input_dim, hidden_size, h_t_enc, c_t_enc, numlayers, dropout_prob_in,
                             dropout_prob, dropout_prob_fc)
            # obtain the loss function
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            count_t = time.time()
        if epoch % 100 == 0:
            print("Epoch: %d, train_loss: %1.5f" % (epoch, loss.item()))
   
    torch.save(seq, "/home/iichsk/workspace/vrep6_lstm_classify/weight/lstmclassify_{}.pt".format(count_t))
    return output


def test(test_data, test_label):
    seq.eval()
    bsz = 1
    
    predictions = np.zeros((int(test_data.shape[1]), output_dim))
    label_class = np.zeros((int(test_label.shape[0]), output_dim))
    z=np.array(range(int(test_data.shape[1]))).reshape(int(test_data.shape[1]))
    optimizer.zero_grad()
    h_t_enc, c_t_enc = init_hidden(bsz, numlayers, hidden_size)  # hidden state와 cell state 초기화
    
    for ii in range(int(test_data.shape[1]/bsz)):
        for i in range(1):
            input_x = torch.Tensor(test_data[:, z[(ii+i)*bsz:(i+1+ii)* bsz], :]).cuda()  # seq_len x bsz x input_dim
            label = torch.Tensor(test_label[z[(i+ii)*bsz:(ii+i+1)* bsz], :]).cuda()
            #test_start = timer() # timer start
            output = seq(input_x, input_dim, hidden_size, h_t_enc, c_t_enc, numlayers, dropout_prob_in,
                                         dropout_prob, dropout_prob_fc)
            #test_end = timer() # timer start
            #test_time = str(test_end-test_start)
            #print('testing loading time = ', test_time)
            #loss = criterion(output, label)
            # obtain the loss function
        predictions[ii:ii+1,:] = output.detach().cpu().data.numpy() #모든값 넣기
        label_class[ii:ii+1,:] = label.detach().cpu().data.numpy()

    return predictions, label_class

def main():
############### load dataset ######################
    path1 = '/home/iichsk/workspace/vrep6_lstm_classify/database/normal.csv'
    dataX1, data_label_Y1 = lp.load_data10(path1, seq_len)
    print('data1 load')
    path2 = '/home/iichsk/workspace/vrep6_lstm_classify/database/joint1error.csv'
    dataX2, data_label_Y2 = lp.load_data10(path2, seq_len)
    print('data2 load')
    path3 = '/home/iichsk/workspace/vrep6_lstm_classify/database/joint2error.csv'
    dataX3, data_label_Y3 = lp.load_data10(path3, seq_len)
    print('data3 load')
    path4 = '/home/iichsk/workspace/vrep6_lstm_classify/database/joint3error.csv'
    dataX4, data_label_Y4 = lp.load_data10(path4, seq_len)
    print('data4 load')
    path5 = '/home/iichsk/workspace/vrep6_lstm_classify/database/joint4error.csv'
    dataX5, data_label_Y5 = lp.load_data10(path5, seq_len)
    print('data5 load')
    path6 = '/home/iichsk/workspace/vrep6_lstm_classify/database/joint5error.csv'
    dataX6, data_label_Y6 = lp.load_data10(path6, seq_len)
    print('data6 load')
    
    dataX = np.concatenate((dataX1,dataX2,dataX3,dataX4,dataX5,dataX6), axis=0)
    #dataX = np.concatenate((dataX1,dataX2,dataX6), axis=0)

    data_label_Y = np.concatenate((data_label_Y1,data_label_Y2,data_label_Y3,data_label_Y4,data_label_Y5,data_label_Y6), axis=0)
    #data_label_Y = np.concatenate((data_label_Y1,data_label_Y2,data_label_Y6), axis=0)

    trainX, testX, train_label, test_label = lp.dataset_split2(dataX, data_label_Y)
    print('test_label',test_label.shape)

###########################################################
    train(trainX, train_label)

    #seq = torch.load("/home/iichsk/workspace/vrep6_lstm_classify/weight/lstmclassify_1618367314.6069791.pt")

########################## test ###########################
    #testX=testX[:,0:1,:]
    #test_label=test_label[0:1,:]
    test_start = timer() # timer start
    test_predict, target = test(testX,test_label)
    test_end = timer() # timer end
    test_time = str(test_end-test_start)
    print('testing time = ', test_time)
    classpredict = np.argmax(test_predict, axis=1)
    target = np.argmax(target, axis=1)  # true classes
    #print("target=",target)
    c_names = ['normal', 'errorjoint1','errorjoint2', 'errorjoint3','errorjoint4','errorjoint5']
    #c_names = ['normal', 'errorjoint1','errorjoint5']

    # Classification Report
    print(classification_report(target, classpredict, target_names=c_names))
    # Confusion Matrix
    print(confusion_matrix(target, classpredict))
    # result print
    print('number of Normal data=',len(classpredict[classpredict==0]))
    print('number of errorjoint1 data=',len(classpredict[classpredict==1]))
    print('number of errorjoint2 data=',len(classpredict[classpredict==2]))
    print('number of errorjoint3 data=',len(classpredict[classpredict==3]))
    print('number of errorjoint4 data=',len(classpredict[classpredict==4]))
    print('number of errorjoint5 data=',len(classpredict[classpredict==5]))
    #print('number of errorjoint6 data=',len(classpredict[classpredict==6]))

if __name__ == '__main__':
    main()
