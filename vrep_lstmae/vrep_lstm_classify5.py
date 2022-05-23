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

from keras.utils import to_categorical

import time
import math
from scipy import io


base_rate = 0.01
learning_rate = base_rate
output_dim = 3
input_dim = 6
hidden_size = 256
numlayers = 3
train_dropout = 0.0
dropout_prob_in = 0.25
dropout_prob = 0.25
dropout_prob_fc = 0.25
seq_len = 25
batchnorm_var = 2e-2
num_epochs = 800
bsz = 256
def __init__(self, seq_len=seq_len, bsz=bsz, hidden_size=hidden_size, dropout_prob_in=dropout_prob_in, dropout_prob=dropout_prob, dropout_prob_fc=dropout_prob_fc,
             numlayers=numlayers):
    self.seq_len = seq_len
    self.bsz = bsz
    self.hidden_size = hidden_size
    self.dropout_prob_in = dropout_prob_in
    self.dropout_prob = dropout_prob
    self.dropout_prob_fc = dropout_prob_fc
    self.numlayers = numlayers

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length -1):
        if((i)%25==0):
            _x = data[i:(i + seq_length-1)]
            _y = data[i]
            x.append(_x)
            y.append(_y)

    return np.array(x), np.array(y)

def sliding_windows2(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length -1):
        if((i)%300==0):
            _x = data[i:(i + seq_length-1)]
            _y = data[i]
            x.append(_x)
            y.append(_y)
    return np.array(x), np.array(y)
#normal data set load
dataset = pd.read_csv('/home/iichsk/workspace/dataset/vrep_dataset/vrep_joint2_normal_1seq2.csv')
train_test_dataset = dataset.filter(
    items=['path', 'seqNum', 'joint1', 'joint2','joint1_difference','joint2_difference'])
label_set = dataset.filter(items=['class'])
le = LabelEncoder()
#label_set_en = le.fit_transform(label_set)
#label_set_en = to_categorical(label_set_en)

train_label_set = label_set[:]
training_set = train_test_dataset.iloc[:, :]

sc = MaxAbsScaler(1)
training_set = sc.fit_transform(training_set)

train_label_Y = np.array(train_label_set)  
#training_set = np.array(training_set)  

train_x, _ = sliding_windows(training_set, seq_len + 1)
_, train_label_Y = sliding_windows(train_label_Y, seq_len + 1)
trainX = np.array(train_x)  # 4428x168x1

#trainX = np.swapaxes(trainX, 0, 1)  # 168 x 4428 x 1


#joint1 error data set load
dataset2 = pd.read_csv('/home/iichsk/workspace/dataset/vrep_dataset/vrep_joint1_bsf_01_1seq2.csv')
train_test_dataset2 = dataset2.filter(
    items=['path', 'seqNum', 'joint1', 'joint2','joint1_difference','joint2_difference'])
label_set2 = dataset2.filter(items=['class'])
train_label_set2 = label_set2[:]

#label_set_en2 = le.fit_transform(label_set2)
#label_set_en2 = to_categorical(label_set_en2)
training_set2 = train_test_dataset2.iloc[:, :]
training_set2 = sc.fit_transform(training_set2)
training_set2_temp = np.zeros(((1450,input_dim)))
train_label_Y2 = np.array(train_label_set2)  
train_label_Y2_temp = np.zeros(((1450,1)))
#training_set2 = np.array(training_set2)  

iii = 0
for i in range(0,17400):
    if i%300==0:
        for ii in range(25):
            a=i
            train_label_Y2_temp[iii+ii:iii+ii+1] = (train_label_Y2[a+ii:a+ii+1])
            training_set2_temp[iii+ii:iii+ii+1,:]= (training_set2[a+ii:a+ii+1,:])
        ii+=1
        iii += ii
training_set2 = training_set2_temp
print(training_set2)
train_label_Y2 = train_label_Y2_temp
print(train_label_Y2)

train_x2, _ = sliding_windows(training_set2, seq_len + 1)

_, train_label_Y2 = sliding_windows(train_label_Y2, seq_len + 1)
trainX2 = np.array(train_x2)  # 4428x168x1
#trainX2 = np.swapaxes(trainX2, 0, 1)  # 168 x 4428 x 1

#joint2 normal data set load
dataset3 = pd.read_csv('/home/iichsk/workspace/dataset/vrep_dataset/vrep_joint2_error_03_1seq2.csv')
train_test_dataset3 = dataset3.filter(
    items=['path', 'seqNum', 'joint1', 'joint2','joint1_difference','joint2_difference'])
label_set3 = dataset3.filter(items=['class'])
train_label_set3 = label_set3[:]
training_set3 = train_test_dataset3.iloc[:, :]
training_set3_len = len(training_set3)


training_set3 = sc.fit_transform(training_set3)

train_label_Y3 = np.array(train_label_set3)  
#training_set3 = np.array(training_set3)  

train_x3, _ = sliding_windows(training_set3, seq_len + 1)

_, train_label_Y3 = sliding_windows(train_label_Y3, seq_len + 1)
trainX3 = np.array(train_x3)  # 4428x168x1
#trainX3 = np.swapaxes(trainX3, 0, 1)  # 168 x 4428 x 1
#concatenate data
trainX = np.concatenate((trainX,trainX2,trainX3), axis=0)
train_label_Y = np.concatenate((train_label_Y, train_label_Y2, train_label_Y3), axis=0)
i_labels = le.fit_transform(train_label_Y)
train_label_Y = to_categorical(i_labels)


trainX, testX, train_label_Y, test_label_Y = train_test_split(trainX, train_label_Y, stratify=i_labels,
                                                    test_size=0.2, random_state = 42)

#testX_invers=sc.inverse_transform(testX[171,:,:])

trainX = np.swapaxes(trainX, 0, 1)  # 168 x 4428 x 1
testX = np.swapaxes(testX, 0, 1)  # 168 x 4428 x 1

# 입력 신호 구조 : seq_len x batch x input_dim

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
        #print("h_t_enc type:", type(h_t_enc))

    return h_t_enc, c_t_enc


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return list(repackage_hidden(v) for v in h)


def loss_function(recon_x, x):
    """MSE를 계산하는 는 함수

    :param  recon_x: Reconstruction 된 데이터
    :param  x: 원래의 input 데이터

    :return:  타입의 계산된 MSE
    """
    MSE = torch.mul((recon_x - x), (recon_x - x))
    MSE = torch.mean(MSE)
    return MSE

criterion = nn.MultiLabelSoftMarginLoss()


def initial_module(self, input_dim, output_dim, enum_min, enum_max, enum_inx, select_input_type):
    """ sequence 함수 초기화

    """

    self.input_dim = input_dim
    self.output_dim = output_dim
    self.select_input_type = select_input_type
    self.seq = Sequence(self.input_dim, self.output_dim, self.hidden_size, self.numlayers, self.dropout_prob_in,
                        self.dropout_prob, self.dropout_prob_fc)
    self.seq.cuda()
    self.seq = torch.nn.DataParallel(self.seq, device_ids=[0])
    self.seq.apply(weights_init)


class Sequence(nn.Module):
    """ LSTM 모델의 구조를 정의 하고, forward 동작에 대해 정의하는  class

    :param int input_dim: 입력 데이터의 dimension
    :param int output_dim: 출력 데이터의 dimension
    :param int hidden_size: LSTM module의 hidden layer의 크기
    :param int numlayers: LSTM module의 layer 수
    :param float dropout_prob_in: 입력 값의 dropout 확률 값 (default: 0.0)
    :param float dropout_prob: LSTM module에 적용되는 dropout 확률 값
    :param float dropout_prob_fc: fully connected layer에 적용되는 dropout 확률 값

    """

    def __init__(self, input_dim, output_dim, hidden_size, numlayers, dropout_prob_in, dropout_prob, dropout_prob_fc):
        super(Sequence, self).__init__()

        enccell_list = []
        enccell_list.append(nn.LSTMCell(input_dim, hidden_size))
        
        for idcell in range(1, int(numlayers)):
            enccell_list.append(nn.LSTMCell(hidden_size, hidden_size))

        self.enccell_list = nn.ModuleList(enccell_list)  # 다중 layer로 구성된 LSTM module

        self.fc_enc = nn.Linear(hidden_size, output_dim)  # fully connected layer

    def forward(self, input, input_dim, hidden_size, h_t_enc, c_t_enc, numlayers, dropout_prob_in, dropout_prob,
                dropout_prob_fc):

        output = []
        mask_list = []
        if self.training == True:  # 훈련에 사용될 dropout 용 mask
            input_mask = torch.bernoulli(
                (1.0 - dropout_prob_in) * torch.ones(input.shape[1], (input_dim + 2 * hidden_size))).cuda() / (
                                 1.0 - dropout_prob_in)
            for ii in range(1, numlayers):
                mask_list.append(
                    torch.bernoulli((1.0 - dropout_prob) * torch.ones(input.shape[1], 3 * hidden_size)).cuda() / (
                            1.0 - dropout_prob))
            fc_mask = torch.bernoulli((1.0 - dropout_prob_fc) * torch.ones(input.shape[1], hidden_size)).cuda() / (
                    1.0 - dropout_prob_fc)
            
        else:  # 테스트에서는 dropout을 사용하지 않기 때문에 mask를 모두 1로 채움
            input_mask = torch.ones(input.shape[1], (input_dim + 2 * hidden_size)).cuda()
            for ii in range(1, numlayers):
                mask_list.append(torch.ones(input.shape[1], 3 * hidden_size).cuda())
            fc_mask = torch.ones(input.shape[1], hidden_size).cuda()

        for i, input_t in enumerate(input.chunk(input.size(0), dim=0)):  # seq len만큼 수행함
            
            h_t_enc[0], c_t_enc[0] = self.enccell_list[0](input_mask[:, :input_dim] * input_t.squeeze(0), (
            input_mask[:, input_dim:input_dim + hidden_size] * h_t_enc[0],
            input_mask[:, input_dim + hidden_size:] * c_t_enc[0]))

            for ii in range(1, numlayers):  # 아래 layer의 LSTM module 출력을 윗 layer의 LSTM module 입력으로 사용
                h_t_enc[ii], c_t_enc[ii] = self.enccell_list[ii](mask_list[ii - 1][:, :hidden_size] * h_t_enc[ii - 1], (
                mask_list[ii - 1][:, hidden_size:2 * hidden_size] * h_t_enc[ii],
                mask_list[ii - 1][:, 2 * hidden_size:] * c_t_enc[ii]))
            output = self.fc_enc(fc_mask * h_t_enc[-1])  # fully-connected layer
            
            #outputs += [output]

        #outputs = torch.stack(outputs, 0)

        return output


seq = Sequence(input_dim=input_dim, output_dim=output_dim, hidden_size=hidden_size, numlayers=numlayers, dropout_prob_in=dropout_prob_in, dropout_prob=dropout_prob, dropout_prob_fc=dropout_prob_fc)
seq.cuda()
# seq = torch.nn.DataParallel(seq, device_ids=[0])
seq.apply(weights_init)

# 입력 신호 구조 : seq_len x batch x input_dim

# trainX 4428


optimizer = torch.optim.Adam(seq.parameters(), lr=learning_rate)


# optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

# Train the model
def train(train_data,train_label):
    seq.train()
    
    for epoch in range(num_epochs):

        learning_rate = 0.003 * pow(0.2, math.floor(float(epoch) / 10.0))
        
        
        z = np.random.permutation(train_data.shape[1])
       
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        for i in range(int(train_data.shape[1] / bsz)):
            optimizer.zero_grad()

            h_t_enc, c_t_enc = init_hidden(bsz, numlayers, hidden_size)  # hidden state와 cell state 초기화

            # LSTM 네트워크에 입력신호를 입력하여 출력신호와 hidden state, cell state를 출력
            input_x = torch.Tensor(train_data[:, z[i*bsz:(i+1)* bsz], :]).cuda()  # seq_len x bsz x input_dim
            #print("train_input:", input_x)
            #print("train_input_x.size():",input_x.size())
            
            label = torch.Tensor(train_label[z[i*bsz:(i+1)* bsz], :]).cuda()
            output = seq(input_x, input_dim, hidden_size, h_t_enc, c_t_enc, numlayers, dropout_prob_in,
                             dropout_prob, dropout_prob_fc)
            # obtain the loss function

            loss = criterion(output, label)
            loss.backward()

            optimizer.step()
            count_t = time.time()

        """print("out=", out)
        print("h_t_enc=", h_t_enc)
        print("c_t_enc=", c_t_enc)"""
        if epoch % 10 == 0:
            print("Epoch: %d, train_loss: %1.5f" % (epoch, loss.item()))
        # train_loss_out_path = "data/loss_epoch_{}_{}.mat".format(epoch, count_t)
        # data = {'lose': loss}
        # io.savemat(train_loss_out_path, data)
    #print("trainoutput=", output)

    return output


def test():
    seq.eval()
    bsz = 1
    
    predictions = np.zeros((int(testX.shape[1]), output_dim))
    label_class = np.zeros((int(test_label_Y.shape[0]), output_dim))

    z=np.array(range(int(testX.shape[1]))).reshape(int(testX.shape[1]))
   
    optimizer.zero_grad()

    h_t_enc, c_t_enc = init_hidden(bsz, numlayers, hidden_size)  # hidden state와 cell state 초기화

    # LSTM 네트워크에 입력신호를 입력하여 출력신호와 hidden state, cell state를 출력
    
    for ii in range(int(testX.shape[1]/bsz)):
        
    
        for i in range(1):
            input_x = torch.Tensor(testX[:, z[(ii+i)*bsz:(i+1+ii)* bsz], :]).cuda()  # seq_len x bsz x input_dim
            label = torch.Tensor(test_label_Y[z[(i+ii)*bsz:(ii+i+1)* bsz], :]).cuda()
            output = seq(input_x, input_dim, hidden_size, h_t_enc, c_t_enc, numlayers, dropout_prob_in,
                                         dropout_prob, dropout_prob_fc)
            loss = criterion(output, label)
            # obtain the loss function

        predictions[ii:ii+1,:] = output.detach().cpu().data.numpy() #모든값 넣기
        label_class[ii:ii+1,:] = label.detach().cpu().data.numpy()

    print("Epoch: , eval_loss: %1.5f" % (loss.item()))

    return predictions, label_class

train(trainX,train_label_Y)
'''train(trainX3,train_label_Y3)
train(trainX,train_label_Y1)
train(trainX2,train_label_Y2)
'''
test_predict, target = test()
print("classpredict=",test_predict.shape)

classpredict = np.argmax(test_predict, axis=1)
print("classpredict=",classpredict)
print('target.shape',target.shape)
target = np.argmax(target, axis=1)  # true classes


print("target=",target)

c_names = ['normal', 'errorjoint1','errorjoint2']


# Classification Report
print(classification_report(target, classpredict, target_names=c_names))
# Confusion Matrix
print(confusion_matrix(target, classpredict))