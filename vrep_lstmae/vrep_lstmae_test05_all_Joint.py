from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import torch
import torch.nn as nn
import sys
from os.path import dirname

sys.path.append(dirname(__file__))
# from torch.autograd import Variable
import time
import math
from scipy import io
#import scipy.stats as sps
import argparse
import os
import glob

from lstm_ae_net import Sequence2
from lstm_ae_preprocessing import lstm_ae_preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
#rv = sps.norm()
num_epochs = 100
base_rate = 0.01
learning_rate = base_rate
bsz = 128
output_dim = 6  # 1
input_dim = 6  # 1
hidden_size = 128
numlayers = 1  # 3
dropout_prob_in = 0.0
dropout_prob = 0.0
dropout_prob_fc = 0.0
seq_len = 5 
batchnorm_var = 2e-2
conditional = False


parser = argparse.ArgumentParser(description='resume training')
parser.add_argument('--epoch', type = int, default = 0, help ="the value of epoch means which traning epoch you want to employ to resume (default:0)")
args = parser.parse_args()
sc = MinMaxScaler()


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
    h_t_dec = []
    c_t_dec = []

    for ii in range(numlayers):
        h_t_enc.append((torch.zeros(bsz, hidden_size, requires_grad=False).cuda()))
        c_t_enc.append((torch.zeros(bsz, hidden_size, requires_grad=False).cuda()))
        h_t_dec.append((torch.zeros(bsz, hidden_size, requires_grad=False).cuda()))
        c_t_dec.append((torch.zeros(bsz, hidden_size, requires_grad=False).cuda()))

    return h_t_enc, c_t_enc, h_t_dec, c_t_dec


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
    """print('recon_X',recon_x)

    recon_x_tmp = (recon_x * 10000)
    print('recon_X * 10000',recon_x_tmp)

    recon_x = torch.round(recon_x_tmp).inteager()
    print('recon_X_round',recon_x)

    recon_x = (recon_x / 10000).float()
    print('recon_X/10000',recon_x)"""
    MSE = torch.mul((recon_x - x), (recon_x - x))  # 32x6
    MSE = torch.mean(MSE)  # 1???
    return MSE


def loss_function_eval(recon_x, x):
    """MSE를 계산하는 는 함수

    :param  recon_x: Reconstruction 된 데이터
    :param  x: 원래의 input 데이터

    :return:  타입의 계산된 MSE
    """
    MSE = torch.mul((recon_x - x), (recon_x - x))  # 32 x 6

    return MSE


def loss_function_test(recon_x, x):
    """test시 MSE를 계산하는 는 함수

    :param  recon_x: Reconstruction 된 데이터
    :param  x: 원래의 input 데이터

    :return: 계산된 MSE
    """
    # recon_x = recon_x.clone()
    # x = x.clone()
    MSE = torch.mul((recon_x - x), (recon_x - x))
    # MSE = torch.mean(MSE,2, keepdim=True).squeeze(2)
    MSE = torch.mean(MSE, 0, keepdim=True).squeeze(0)

    return MSE


def anomaly_score(recon_x, swap_x, x):
    S = 1 - ((torch.mm(recon_x, swap_x)) / torch.mul((torch.norm(x)), torch.norm(x)))  # 1x1

    return S


def anomaly_likelyhood(S_t, mu_t_hat):
    # print('S_t',S_t)
    mu_t = np.mean(S_t, 0)
    # print('mu_t_hat',mu_t_hat)
    sigma_t = np.std(S_t, 0)
    # print('sigma_t',sigma_t)

    x = (mu_t_hat - mu_t) / sigma_t
    x = x / (2 ** 0.5)
    qfunc = 0.5 * math.erfc(x)
    L_t = 1 - qfunc
    return L_t, qfunc




seq = Sequence2(input_dim, output_dim, hidden_size, numlayers, dropout_prob_in, dropout_prob, dropout_prob_fc,
                conditional)
seq.cuda()
# seq = torch.nn.DataParallel(seq, device_ids=[0])


# resume train, save trained weights of seq
# fc_horse.py 73-75 173-187 390-420

if args.epoch != 0:
    tmp = glob.glob("data/model_epoch_{}_*.pth".format(args.epoch))
    print("tmp len", len(tmp))
    if len(tmp) == 1:
        print("len is 1")
        check_point = torch.load(tmp[0])
        # config = check_point['config']
        args.epoch = check_point['epoch']
        output = check_point['output']
        seq.load_state_dict(check_point['state_dict'])
        del check_point
    elif len(tmp) > 1:
        print("there is more than two pth files for {} epoch".format(args.epoch))
        sys.exit(1)
    elif len(tmp) == 0:
        print("there is no pth file for {} epoch".format(args.epoch))
        sys.exit(1)
else:
    seq.apply(weights_init)
    print("len is 0", args.epoch)
# seq.apply(weights_init)


# 입력 신호 구조 : seq_len x batch x input_dim



optimizer = torch.optim.Adam(seq.parameters(), lr=learning_rate)



# Train the model
def train(trainX, epoch, learning_rate, bsz,  input_dim, hidden_size, numlayers, dropout_prob_in,
          dropout_prob, dropout_prob_fc):
    seq.train()

    learning_rate = learning_rate * pow(0.25, math.floor(float(epoch - 1) / 20.0))

    z = np.random.permutation(trainX.shape[1])

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    acc_loss = np.zeros((int(trainX.shape[1] / bsz), 1))

    for i in range(int(trainX.shape[1] / bsz)):
        optimizer.zero_grad()

        h_t_enc, c_t_enc, h_t_dec, c_t_dec = init_hidden(bsz, numlayers, hidden_size)  # hidden state와 cell state 초기화

        # LSTM 네트워크에 입력신호를 입력하여 출력신호와 hidden state, cell state를 출력

        input_x = (torch.Tensor(trainX[:, z[i * bsz:(i + 1) * bsz], :]).cuda()).float()  # seq_len x bsz x input_dim
        # print('input=',input_x)
        input_r_tmp = np.flip(trainX[:, z[i * bsz:(i + 1) * bsz], :], axis=0).copy()
        input_r = (torch.Tensor(input_r_tmp).cuda()).float()
        # target = torch.Tensor(trainY[z[i*bsz:(i+1)* bsz], :]).cuda()
        output, _, _, _, _ = seq(input_x, input_dim, hidden_size, h_t_enc, c_t_enc, numlayers, dropout_prob_in,
                                 dropout_prob, dropout_prob_fc, input_r, h_t_dec, c_t_dec, conditional)
        # obtain the loss function
        # print('output=',output)
        loss = loss_function(output, input_r)

        loss.backward()

        optimizer.step()
        acc_loss[i, 0] = loss
        # print('acc',acc_loss)
    return acc_loss


def evaluate(testX, bsz, output_dim, input_dim, hidden_size, numlayers, dropout_prob_in, dropout_prob, dropout_prob_fc):
    seq.eval()

    predictions = np.zeros((seq_len, int(testX.shape[1]), output_dim))
    loss = np.zeros((int(testX.shape[1]), output_dim))
    S_t = np.zeros((int(testX.shape[1])))
    mu_t_hat = np.zeros((int(testX.shape[1])))
    a_likelyhood = np.zeros((int(testX.shape[1])))
    # LSTM 네트워크에 입력신호를 입력하여 출력신호와 hidden state, cell state를 출력
    for ii in range(int(testX.shape[1] / bsz)):
        input_x = torch.Tensor(testX[:, ii * bsz:(ii + 1) * bsz, :]).cuda()  # seq_len x bsz x input_dim
        input_r_tmp = np.flip(testX[:, ii * bsz:(ii + 1) * bsz, :], axis=0).copy()
        input_r = torch.Tensor(input_r_tmp).cuda()
        #input_r_swap = torch.Tensor(np.swapaxes(input_r_tmp, 1, 2)).cuda()

        h_t_enc, c_t_enc, h_t_dec, c_t_dec = init_hidden(bsz, numlayers, hidden_size)  #
        output, _, _, _, _ = seq(input_x, input_dim, hidden_size, h_t_enc, c_t_enc, numlayers, dropout_prob_in,
                                 dropout_prob, dropout_prob_fc, input_r, h_t_dec, c_t_dec, conditional)
        predictions[:, ii * bsz:(ii + 1) * bsz, :] = output.cpu().data.numpy()

        # obtain the loss function
        tmp = loss_function_test(output, input_r)
        loss[ii * bsz:(ii + 1) * bsz, :] = tmp.cpu().data.numpy()  # 32 x output_dim
        '''S_t[ii * bsz] = (
                                    anomaly_score(output[seq_len - 1, :, :], input_r_swap[seq_len - 1, :, :], input_r[seq_len - 1, :, :]))
                                if (ii + 1) % 2 == 0 and ii != 0:
                                    mu_t_hat[(ii - 1) * bsz:(ii) * bsz] = np.mean(S_t[(ii - 1) * bsz:(ii) * bsz], 0)
                                    # print('mu_t_hat',mu_t_hat)
                        
                            for iii in range(ii + 1):
                                a_likelyhood[iii], _ = anomaly_likelyhood(S_t, mu_t_hat[iii])'''
    predictions = np.flip(predictions, axis=0).copy()
    #a_likelyhood = np.flip(a_likelyhood, axis=0).copy()

    return predictions, loss


def checkpoint2():
    count_t = time.time()
    model_out_dir = "data_sql_{}_bsz_{}_hidden_{}".format(seq_len, bsz, hidden_size)

    if not os.path.isdir(model_out_dir):
        os.mkdir(model_out_dir)

    model_out_path = "weight/model_epoch_{}_{}.pth".format(epoch, count_t)
    torch.save({
        'state_dict': seq.state_dict(),
    }, model_out_path)
    dropoutlabel = int(dropout_prob * 1000)
    print("dropoutlabel=", dropoutlabel)
    print("conf_evaluate saved to data_sql_{}_bsz_{}_hidden_{}\n".format(seq_len, bsz, hidden_size))

    io.savemat(
        "data_sql_{}_bsz_{}_hidden_{}/alljoint_dropout_{}_epoch_{}_lr_{}_{}.mat".format(
            seq_len, bsz, hidden_size, dropout_prob, epoch, learning_rate, count_t),
        {'joint1normalpredict': predict1_normal,'joint1normaltarget': normal_inverse1,
        'joint2normalpredict': predict2_normal, 'joint2normaltarget': normal_inverse2, 
        'joint1errorpredict': predict3_error, 'joint2errorpredict': predict4_error,
         'trainloss': train_loss, })

    return

#dta load
lap = lstm_ae_preprocessing()
path = '/home/iichsk/workspace/dataset/vrep_dataset/vrep_joint1_normal.csv'
trainX_tmp1, testX_tmp1=lap.load_data(path, seq_len)

path = '/home/iichsk/workspace/dataset/vrep_dataset/vrep_joint2_normal.csv'
trainX_tmp2, testX_tmp2=lap.load_data(path, seq_len)

trainX = np.concatenate((trainX_tmp1,trainX_tmp2),axis=1)
testX = np.concatenate((testX_tmp1,testX_tmp2),axis=1)

print('trainX.shape======',trainX.shape)

#train start
print('start train....')
print('trainX_tmp1',trainX_tmp1.shape)
for epoch in range(args.epoch + 1, num_epochs + 1):
    train_loss = train(trainX_tmp1, epoch, learning_rate, bsz, input_dim, hidden_size, numlayers,
                               dropout_prob_in, dropout_prob, dropout_prob_fc)
    if (epoch + 1) % 1 == 0:
            print("Epoch: %d, train_loss: %.3e" % (epoch, np.mean(train_loss)))

print('start train2....')
print('trainX_tmp2',trainX_tmp2.shape)
for epoch in range(args.epoch + 1, num_epochs + 1):
    train_loss = train(trainX_tmp2, epoch, learning_rate, bsz, input_dim, hidden_size, numlayers,
                               dropout_prob_in, dropout_prob, dropout_prob_fc)
    if (epoch + 1) % 1 == 0:
            print("Epoch: %d, train_loss: %.3e" % (epoch, np.mean(train_loss)))
#test start
print('start joint1 test....')
test_bsz=1
predict1, eval_loss1 = evaluate(testX_tmp1, test_bsz, output_dim, input_dim, hidden_size, numlayers,
                                                     dropout_prob_in, dropout_prob, dropout_prob_fc)
predict1_normal, normal_inverse1 = lap.normalize_inverse(predict1, testX_tmp1)
print('normal joint1 data test loss = ', eval_loss1.mean())

print('start joint2 test....')
test_bsz=1
predict2, eval_loss2 = evaluate(testX_tmp2, test_bsz, output_dim, input_dim, hidden_size, numlayers,
                                                     dropout_prob_in, dropout_prob, dropout_prob_fc)
predict2_normal, normal_inverse2 = lap.normalize_inverse(predict2, testX_tmp2)
print('normal joint1 data test loss = ', eval_loss2.mean())

print('start joint1 error test....')
test_bsz=1
path2='/home/iichsk/workspace/dataset/vrep_dataset/vrep_joint1_bsf_01.csv'
_, error_joint1_testX=lap.load_data(path2, seq_len)
predict3, eval_loss3 = evaluate(error_joint1_testX, test_bsz, output_dim, input_dim, hidden_size, numlayers,
                                                     dropout_prob_in, dropout_prob, dropout_prob_fc)
print('error joint1 data test loss = ', eval_loss3.mean())

predict3_error, _ = lap.normalize_inverse(predict3, error_joint1_testX)

print('start joint2 error test....')
test_bsz=1
path2='/home/iichsk/workspace/dataset/vrep_dataset/vrep_joint2_error_03.csv'
_, error_joint2_testX=lap.load_data(path2, seq_len)
predict4, eval_loss4 = evaluate(error_joint2_testX, test_bsz, output_dim, input_dim, hidden_size, numlayers,
                                                     dropout_prob_in, dropout_prob, dropout_prob_fc)
print('error joint2 data test loss = ', eval_loss4.mean())

predict4_error, _ = lap.normalize_inverse(predict4, error_joint2_testX)

'''x = np.arange(0,int(predict2_inverse.shape[1]))
y1 = predict2_inverse[0, :, 5]
y2 = testX2_inverse[0, :, 5]

x2 = np.arange(0,int(predict1_normal.shape[1]))
y3 = predict1_normal[0, :, 5]
y4 = normal_inverse[1, :, 5]
y5 = y3-y1
predict_target = y3-y4
y7 = y1-y4
print('y3',y3)
print('y4',y4)'''
'''print('drawing figure.......')
fig1=plt.figure(1)
plt.plot(x, y1)

plt.title('joint2 Error')
plt.xlabel('sequence')
plt.ylabel('joint2 position value')
plt.savefig('figure_joint2_error.png')
plt.close(fig1)

print('drawing figure2.......')
fig2=plt.figure(2)
plt.plot(x2, y3)

plt.title('joint2 Normal')
plt.xlabel('sequence')
plt.ylabel('joint2 position value')
plt.savefig('figure_normal_joint2.png')
plt.close(fig2)
'''
'''fig3=plt.figure(3)
plt.plot(x2, y5)

plt.title('Normal-joint1')
plt.xlabel('sequence')
plt.ylabel('joint1 encoder')
plt.savefig('figure_normal_bsf_differ.png')
plt.close(fig3)

fig4=plt.figure(4)
plt.plot(x2, predict_target)
plt.title('predict normal-target normal')
plt.xlabel('sequence')
plt.ylabel('joint2 position value')
plt.savefig('figure_normal_joint2_target_differ.png')
plt.close(fig4)

fig5=plt.figure(5)
plt.plot(x2, y7)
plt.title('joint2 error-target')
plt.xlabel('sequence')
plt.ylabel('joint2 position value')
plt.savefig('figure_bsf_joint2_target_differ.png')
plt.close(fig5)
'''
print('saving_data...')
checkpoint2()
