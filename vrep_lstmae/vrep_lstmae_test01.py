from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
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
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
#rv = sps.norm()
num_epochs = 2
base_rate = 0.01
learning_rate = base_rate
bsz = 128
output_dim = 8  # 1
input_dim = 8  # 1
hidden_size = 128
numlayers = 2  # 3
dropout_prob_in = 0.0
dropout_prob = 0.3
dropout_prob_fc = 0.3
seq_len = 10  
batchnorm_var = 2e-2
conditional = False


parser = argparse.ArgumentParser(description='resume training')
parser.add_argument('--epoch', type = int, default = 0, help ="the value of epoch means which traning epoch you want to employ to resume (default:0)")
args = parser.parse_args()

#create sequence
def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length -1):
        _x = data[i:(i + seq_length-1)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)

#normal data set load
dataset = pd.read_csv('/home/iichsk/workspace/dataset/vrep_dataset/vrep_normal_01.csv')
train_test_dataset = dataset.filter(
    items=['targetJoint1', 'targetJoint2', 'path', 'seqNum', 'joint1', 'joint2', 'joint1_difference', 'joint2_difference'])
label_set = dataset.filter(items=['class'])
le = LabelEncoder()
label_set_en = le.fit_transform(label_set)
label_set_en = to_categorical(label_set_en)

train_label_set = label_set_en[0:14400, :]
test_label_set = label_set_en[14401:20700, :]
training_set = train_test_dataset.iloc[0:14400, :]
test_set = train_test_dataset.iloc[14401:20700, :]
training_set_len = len(training_set)
test_set_len = len(test_set)

sc = MaxAbsScaler(1)
training_set = sc.fit_transform(training_set)
test_set = sc.fit_transform(test_set)

train_label_Y = np.array(train_label_set)  
test_label_Y = np.array(test_label_set) 

train_x, _ = sliding_windows(training_set, seq_len + 1)
test_x, _ = sliding_windows(test_set, seq_len + 1)

_, train_label_Y = sliding_windows(train_label_Y, seq_len + 1)
_, test_label_Y = sliding_windows(test_label_Y, seq_len + 1)
trainX = np.array(train_x)  # 4428x168x1
trainX = np.swapaxes(trainX, 0, 1)  # 168 x 4428 x 1
testX = np.array(test_x) # 6609 x 168 x 1
testX = np.swapaxes(testX, 0, 1) # 10 x size x 8

#joint1 error data set load
train_test_dataset2 = dataset.filter(
    items=['targetJoint1', 'targetJoint2', 'path', 'seqNum', 'joint1', 'joint2', 'joint1_difference', 'joint2_difference'])
train_label_set2 = label_set_en[20701:35100]
test_label_set2 = label_set_en[35101:41400]
training_set2 = train_test_dataset2.iloc[20701:35100, :]
test_set2 = train_test_dataset2.iloc[35101:41400, :]
print('training_set2',training_set2.tail(3))
print('test_set2',test_set2.tail(3))
training_set2_len = len(training_set2)
test_set2_len = len(test_set2)


training_set2 = sc.fit_transform(training_set2)
test_set2 = sc.fit_transform(test_set2)

train_label_Y2 = np.array(train_label_set2)  
test_label_Y2 = np.array(test_label_set2) 

train_x2, _ = sliding_windows(training_set2, seq_len + 1)
test_x2, _ = sliding_windows(test_set2, seq_len + 1)

_, train_label_Y2 = sliding_windows(train_label_Y2, seq_len + 1)
_, test_label_Y2 = sliding_windows(test_label_Y2, seq_len + 1)
trainX2 = np.array(train_x2)  # 4428x168x1
trainX2 = np.swapaxes(trainX2, 0, 1)  # 168 x 4428 x 1
testX2 = np.array(test_x2) # 6609 x 168 x 1
testX2 = np.swapaxes(testX2, 0, 1) # 10 x 6609 x 8

#joint2 normal data set load
train_test_dataset3 = dataset.filter(
    items=['targetJoint1', 'targetJoint2', 'path', 'seqNum', 'joint1', 'joint2', 'joint1_difference', 'joint2_difference'])
train_label_set3 = label_set_en[41401:44900]
test_label_set3 = label_set_en[44901:46375]
training_set3 = train_test_dataset3.iloc[41401:44900, :]
test_set3 = train_test_dataset3.iloc[44901:46375, :]
print('train_set2',training_set3.tail(3))
print('test_set2',test_set3.tail(3))

training_set3_len = len(training_set3)
test_set3_len = len(test_set3)


training_set3 = sc.fit_transform(training_set3)
test_set3 = sc.fit_transform(test_set3)

train_label_Y3 = np.array(train_label_set3)  
test_label_Y3 = np.array(test_label_set3) 

train_x3, _ = sliding_windows(training_set3, seq_len + 1)
test_x3, _ = sliding_windows(test_set3, seq_len + 1)

_, train_label_Y3 = sliding_windows(train_label_Y3, seq_len + 1)
_, test_label_Y3 = sliding_windows(test_label_Y3, seq_len + 1)
trainX3 = np.array(train_x3)  # 4428x168x1
trainX3 = np.swapaxes(trainX3, 0, 1)  # 168 x 4428 x 1
testX3 = np.array(test_x3) # 6609 x 168 x 1
testX3 = np.swapaxes(testX3, 0, 1) # 10 x 6609 x 8

#joint2 normal data set load
train_test_dataset4 = dataset.filter(
    items=['targetJoint1', 'targetJoint2', 'path', 'seqNum', 'joint1', 'joint2', 'joint1_difference', 'joint2_difference'])
train_label_set4 = label_set_en[46376:49875]
test_label_set4 = label_set_en[49876:]
training_set4 = train_test_dataset4.iloc[46376:49875, :]
test_set4 = train_test_dataset4.iloc[49876:, :]
print('train_set4',training_set4.tail(3))
print('test_set4',test_set4.tail(3))

training_set4_len = len(training_set4)
test_set4_len = len(test_set4)


training_set4 = sc.fit_transform(training_set4)
test_set4 = sc.fit_transform(test_set4)

train_label_Y4 = np.array(train_label_set4)  
test_label_Y4 = np.array(test_label_set4) 

train_x4, _ = sliding_windows(training_set4, seq_len + 1)
test_x4, _ = sliding_windows(test_set4, seq_len + 1)
_, train_label_Y4 = sliding_windows(train_label_Y4, seq_len + 1)
_, test_label_Y4 = sliding_windows(test_label_Y4, seq_len + 1)
trainX4 = np.array(train_x4)  # 4428x168x1
trainX4 = np.swapaxes(trainX4, 0, 1)  # 168 x 4428 x 1
testX4 = np.array(test_x4) # 6609 x 168 x 1
testX4 = np.swapaxes(testX4, 0, 1) # 10 x 6609 x 8
#concatenate data
trainX = np.concatenate((trainX,trainX2,trainX3,trainX4), axis=1)
train_label_Y = np.concatenate((train_label_Y, train_label_Y2, train_label_Y3, train_label_Y4), axis=0)

testX = np.concatenate((testX,testX2,testX3,testX4), axis=1)
print('testX',testX.shape)
test_label_Y = np.concatenate((test_label_Y, test_label_Y2, test_label_Y3, test_label_Y4), axis=0)
print('test_label_Y',test_label_Y.shape)



def weights_init(m):
    """pytorch??? network??? ????????? ???????????? ??????f

    :param Object m: ???????????? ?????? ?????? network instance

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
    """LSTM h??? c?????? ????????? ?????? ??????

    :param int bsz: ???????????? batch??? ??????
    :param int numlayers: LSTM??? layer ???
    :param int hidden_size: LSTM??? hidden ??????

    :return: ???????????? h_t_enc, c_t_enc, h_t_dec, c_t_dec
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
    """MSE??? ???????????? ??? ??????

    :param  recon_x: Reconstruction ??? ?????????
    :param  x: ????????? input ?????????

    :return:  ????????? ????????? MSE
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
    """MSE??? ???????????? ??? ??????

    :param  recon_x: Reconstruction ??? ?????????
    :param  x: ????????? input ?????????

    :return:  ????????? ????????? MSE
    """
    MSE = torch.mul((recon_x - x), (recon_x - x))  # 32 x 6

    return MSE


def loss_function_test(recon_x, x):
    """test??? MSE??? ???????????? ??? ??????

    :param  recon_x: Reconstruction ??? ?????????
    :param  x: ????????? input ?????????

    :return: ????????? MSE
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


# ?????? ?????? ?????? : seq_len x batch x input_dim



optimizer = torch.optim.Adam(seq.parameters(), lr=learning_rate)


# optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

# Train the model
def train(trainX, epoch, learning_rate, bsz,  input_dim, hidden_size, numlayers, dropout_prob_in,
          dropout_prob, dropout_prob_fc):
    seq.train()

    learning_rate = learning_rate * pow(0.5, math.floor(float(epoch - 1) / 50.0))

    z = np.random.permutation(trainX.shape[1])

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    acc_loss = np.zeros((int(trainX.shape[1] / bsz), 1))

    for i in range(int(trainX.shape[1] / bsz)):
        optimizer.zero_grad()

        h_t_enc, c_t_enc, h_t_dec, c_t_dec = init_hidden(bsz, numlayers, hidden_size)  # hidden state??? cell state ?????????

        # LSTM ??????????????? ??????????????? ???????????? ??????????????? hidden state, cell state??? ??????

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
    # LSTM ??????????????? ??????????????? ???????????? ??????????????? hidden state, cell state??? ??????
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


'''def evaluate_confidence(bsz, output_dim, input_dim, hidden_size, numlayers, dropout_prob_in, dropout_prob,
                        dropout_prob_fc, num_evals, start_day_tmp):
    seq.train()

    predictions = np.zeros((num_evals, seq_len, int(dataX.shape[1]), output_dim))

    loss = np.zeros((num_evals, int(dataX.shape[1]), output_dim))

    for jj in range(num_evals):

        print('evaluate_confidence_number=', jj)
        # print("prediction=",predictions.shape)
        # LSTM ??????????????? ??????????????? ???????????? ??????????????? hidden state, cell state??? ??????
        for ii in range(int((dataX.shape[1] / bsz))):
            input_x = torch.Tensor(dataX[:, ii * bsz:(ii + 1) * bsz, :]).cuda()  # seq_len x bsz x input_dim
            input_r_tmp = np.flip(dataX[:, ii * bsz:(ii + 1) * bsz, :], axis=0).copy()
            input_r = torch.Tensor(input_r_tmp).cuda()
            input_r_swap = torch.Tensor(np.swapaxes(input_r_tmp, 1, 2)).cuda()

            h_t_enc, c_t_enc, h_t_dec, c_t_dec = init_hidden(bsz, numlayers, hidden_size)  #

            output, _, _, _, _ = seq(input_x, input_dim, hidden_size, h_t_enc, c_t_enc, numlayers, dropout_prob_in,
                                     dropout_prob, dropout_prob_fc, input_r, h_t_dec, c_t_dec, conditional)

            predictions[jj, :, (ii * bsz):((ii + 1) * bsz), :] = output.cpu().data.numpy()  # ????????? ??????
            tmp = loss_function_test(output, input_r)
            loss[jj, ii * bsz:(ii + 1) * bsz, :] = tmp.cpu().data.numpy()  # 64 x output_dim
        

        predictions = np.flip(predictions, axis=1).copy()

    print("evaluate_confidence mean_eval_loss: %.3e" % np.mean(loss))

    return predictions, loss
'''

'''def checkpoint(epoch, loss):
    count_t = time.time()
    model_out_dir = "data"

    if not os.path.isdir(model_out_dir):
        os.mkdir(model_out_dir)

    model_out_path = "data/model_epoch_{}_{}.pth".format(epoch, count_t)
    torch.save({
        'epoch': epoch,
        'state_dict': seq.state_dict(),
    }, model_out_path)

    print("Checkpoint saved to_{}_{}".format(dropout_prob, model_out_path))

    io.savemat("data/confresult_epoch_{}_{}.mat".format(epoch, count_t), {'dataX': dataX, 'trainloss': train_loss})

    return'''


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
        "data_sql_{}_bsz_{}_hidden_{}/dropout_{}_epoch_{}_lr_{}_{}.mat".format(
            seq_len, bsz, hidden_size, dropout_prob, epoch, learning_rate, count_t),
        {'testX': testX_inverse, 'evalpredict': eval_predict_inverse, 
        'MSE{}': eval_loss, 'trainloss': train_loss, 'dataset': train_test_dataset,})

    return
for epoch in range(args.epoch + 1, num_epochs + 1):
    train_loss = train(trainX, epoch, learning_rate, bsz, input_dim, hidden_size, numlayers,
                               dropout_prob_in, dropout_prob, dropout_prob_fc)
    if (epoch + 1) % 1 == 0:
            print("Epoch: %d, train_loss: %.3e" % (epoch, np.mean(train_loss)))

eval_predict, eval_loss = evaluate(testX, bsz, output_dim, input_dim, hidden_size, numlayers,
                                                     dropout_prob_in, dropout_prob, dropout_prob_fc)
eval_predict_tmp = np.zeros((int(len(eval_predict[1])), output_dim))
eval_predict_inverse = np.zeros((seq_len, int(len(eval_predict[1])), output_dim))
testX_tmp = np.zeros((int(len(testX[1])), output_dim))
testX_inverse = np.zeros((seq_len, int(len(testX[1])), output_dim))
for iii in range(seq_len):
    eval_predict_tmp[:, :] = eval_predict[iii, :, :]
    eval_predict_inverse[iii, :, :] = sc.inverse_transform(eval_predict_tmp)
    testX_tmp[:, :] = testX[iii, :, :]
    testX_inverse[iii, :, :] = sc.inverse_transform(testX_tmp)
print('saving_data...')
checkpoint2()
'''for zz in range(predict_day + 1):
    x, _ = sliding_windows(training_data_c, data_set, seq_len + 1)
    print('trdac',len(training_set_c))
    print('x.shape',x.shape)
    start_day_tmp = start_day + zz
    trainX, dataX, train_size = input_data(start_day_tmp, train_day, x)
    for epoch in range(args.epoch + 1, num_epochs + 1):

        train_loss = train(epoch, num_epochs, learning_rate, bsz, output_dim, input_dim, hidden_size, numlayers,
                           dropout_prob_in, dropout_prob, dropout_prob_fc, seq_len, batchnorm_var)

        if (epoch + 1) % 1 == 0:
            print("Epoch: %d, train_loss: %.3e" % (epoch, np.mean(train_loss)))
            checkpoint(epoch, train_loss)
            print('saving checkpoint at Epoch {}\n'.format(epoch))
    bsz = 1
    x, _ = sliding_windows(training_data, data_set, seq_len + 1)


    trainX, dataX, train_size = input_data(start_day, train_day, x)
    print('evaluate start...')
    eval_predict, eval_loss, a_likelyhood = evaluate(bsz, output_dim, input_dim, hidden_size, numlayers,
                                                     dropout_prob_in, dropout_prob, dropout_prob_fc)
    eval_predict_tmp = np.zeros((int(len(eval_predict[1])), output_dim))
    eval_predict_inverse = np.zeros((seq_len, int(len(eval_predict[1])), output_dim))
    dataX_tmp = np.zeros((int(len(dataX[1])), output_dim))
    dataX_inverse = np.zeros((seq_len, int(len(dataX[1])), output_dim))

    print('confidence evaluate start...')

    for iii in range(seq_len):
        eval_predict_tmp[:, :] = eval_predict[iii, :, :]
        eval_predict_inverse[iii, :, :] = sc.inverse_transform(eval_predict_tmp)
        dataX_tmp[:, :] = dataX[iii, :, :]
        dataX_inverse[iii, :, :] = sc.inverse_transform(dataX_tmp)

    if epoch == num_epochs:
        bsz = 1
        conf_eval_predict, conf_loss = evaluate_confidence(bsz, output_dim, input_dim, hidden_size,
                                                           numlayers, dropout_prob_in, dropout_prob,
                                                           dropout_prob_fc, num_evals, start_day_tmp)
        conf_eval_predict_tmp = np.zeros((seq_len, int(conf_eval_predict.shape[2]), output_dim))
        conf_eval_predict_tmp2 = np.zeros((int(conf_eval_predict.shape[2]), output_dim))
        conf_eval_predict_inverse = np.zeros((seq_len, int(conf_eval_predict.shape[2]), output_dim))
        conf_eval_predict_inverse_final = np.zeros((num_evals, seq_len, int(conf_eval_predict.shape[2]), output_dim))
        for num in range(num_evals):
            conf_eval_predict_tmp[:, :, :] = conf_eval_predict[num, :, :, :]
            for zzzz in range(seq_len):
                conf_eval_predict_tmp2[:, :] = conf_eval_predict_tmp[zzzz, :, :]
                conf_eval_predict_inverse[zzzz, :, :] = sc.inverse_transform(conf_eval_predict_tmp2)
            conf_eval_predict_inverse_final[num, :, :, :] = conf_eval_predict_inverse
        actual_interval = np.zeros((100, 1))

        conf_lower_lim60 = np.zeros((seq_len, int(conf_eval_predict_inverse_final.shape[2]), output_dim))
        conf_upper_lim60 = np.zeros((seq_len, int(conf_eval_predict_inverse_final.shape[2]), output_dim))
        conf_lower_lim80 = np.zeros((seq_len, int(conf_eval_predict_inverse_final.shape[2]), output_dim))
        conf_upper_lim80 = np.zeros((seq_len, int(conf_eval_predict_inverse_final.shape[2]), output_dim))
        conf_lower_lim95 = np.zeros((seq_len, int(conf_eval_predict_inverse_final.shape[2]), output_dim))
        conf_upper_lim95 = np.zeros((seq_len, int(conf_eval_predict_inverse_final.shape[2]), output_dim))

        # plot mean, mean+2*std, mean-2*std
        print('calculate ci....')
        ci = 0.00
        for z in range(100):
            lower_lim = np.quantile(conf_eval_predict_inverse_final, 0.5 - ci / 2, axis=0)
            upper_lim = np.quantile(conf_eval_predict_inverse_final, 0.5 + ci / 2, axis=0)

            if z == 60:
                conf_lower_lim60 = lower_lim
                conf_upper_lim60 = upper_lim
            elif z == 80:
                conf_lower_lim80 = lower_lim
                conf_upper_lim80 = upper_lim
            elif z == 95:
                conf_lower_lim95 = lower_lim
                conf_upper_lim95 = upper_lim

            actual_interval[ci] = (
                        len(dataX[(dataX <= upper_lim[167, :, :]) & (dataX >= lower_lim[167, :, :])]) / len(dataX))
            ci = ci + 0.01

        mean_prediction = np.mean(conf_eval_predict_inverse_final, 0)
        std_prediction = np.std(conf_eval_predict_inverse_final, 0)
        mean_loss = np.mean(conf_loss, 0)
        lendata = len(x)
        print('saving_data...')
        checkpoint2(epoch, conf_loss, conf_eval_predict)
'''

