from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn



class Sequence2(nn.Module):
    """ LSTM 모델의 구조를 정의 하고, forward 동작에 대해 정의하는  class

    :param int input_dim: 입력 데이터의 dimension
    :param int output_dim: 출력 데이터의 dimension
    :param int hidden_size: LSTM module의 hidden layer의 크기
    :param int numlayers: LSTM module의 layer 수
    :param float dropout_prob_in: 입력 값의 dropout 확률 값 (default: 0.0)
    :param float dropout_prob: LSTM module에 적용되는 dropout 확률 값
    :param float dropout_prob_fc: fully connected layer에 적용되는 dropout 확률 값

    """

    def __init__(self, input_dim, output_dim, hidden_size, numlayers, dropout_prob_in, dropout_prob, dropout_prob_fc,
                 conditional):
        super(Sequence2, self).__init__()

        enccell_list = []
        enccell_list.append(nn.LSTMCell(input_dim, hidden_size))

        for idcell in range(1, int(numlayers)):
            enccell_list.append(nn.LSTMCell(hidden_size, hidden_size))

        self.enccell_list = nn.ModuleList(enccell_list)  # 다중 layer로 구성된 LSTM module
        self.fcnorm_enc = nn.Linear(hidden_size, input_dim, bias=False)
        self.batchnorm_enc = nn.BatchNorm1d(input_dim)

        self.fc_enc = nn.Linear(hidden_size, input_dim)  # fully connected layer

        deccell_list = []
        deccell_list.append(
            nn.LSTMCell(input_dim, hidden_size))  # the first one has a different number of input channels

        for idcell in range(1, numlayers):
            deccell_list.append(nn.LSTMCell(hidden_size, hidden_size))

        self.deccell_list = nn.ModuleList(deccell_list)

        self.fcnorm_dec = nn.Linear(hidden_size, input_dim, bias=False)
        self.batchnorm_dec = nn.BatchNorm1d(input_dim)
        self.fc_dec = nn.Linear(hidden_size, input_dim)
        # self.dropout_fc_dec = nn.Dropout(p = dropout_prob_fc)

    def forward(self, input, input_dim, hidden_size, h_t_enc, c_t_enc, numlayers, dropout_prob_in, dropout_prob,
                dropout_prob_fc, input_r, h_t_dec, c_t_dec, conditional):

        outputs = []
        mask_list = []
        mask_list2 = []

        if self.training == True:  # 훈련에 사용될 dropout 용 mask
            input_mask = torch.bernoulli(
                (1.0 - dropout_prob_in) * torch.ones(input.shape[1], (input_dim + 2 * hidden_size))).cuda() / (
                                 1.0 - dropout_prob_in)
            input_mask2 = torch.bernoulli(
                (1.0 - dropout_prob_in) * torch.ones(input.shape[1], (input_dim + 2 * hidden_size))).cuda() / (
                                  1.0 - dropout_prob_in)
            for ii in range(1, numlayers):
                mask_list.append(
                    torch.bernoulli((1.0 - dropout_prob) * torch.ones(input.shape[1], 3 * hidden_size)).cuda() / (
                            1.0 - dropout_prob))
                mask_list2.append(
                    torch.bernoulli((1.0 - dropout_prob) * torch.ones(input.shape[1], 3 * hidden_size)).cuda() / (
                            1.0 - dropout_prob))
            fc_mask = torch.bernoulli((1.0 - dropout_prob_fc) * torch.ones(input.shape[1], hidden_size)).cuda() / (
                    1.0 - dropout_prob_fc)
            fc_mask2 = torch.bernoulli((1.0 - dropout_prob_fc) * torch.ones(input.shape[1], hidden_size)).cuda() / (
                    1.0 - dropout_prob_fc)
        else:  # 테스트에서는 dropout을 사용하지 않기 때문에 mask를 모두 1로 채움
            input_mask = torch.ones(input.shape[1], (input_dim + 2 * hidden_size)).cuda()
            input_mask2 = torch.ones(input.shape[1], (input_dim + 2 * hidden_size)).cuda()
            for ii in range(1, numlayers):
                mask_list.append(torch.ones(input.shape[1], 3 * hidden_size).cuda())
                mask_list2.append(torch.ones(input.shape[1], 3 * hidden_size).cuda())
            fc_mask = torch.ones(input.shape[1], hidden_size).cuda()
            fc_mask2 = torch.ones(input.shape[1], hidden_size).cuda()

        # enc
        for i, input_t in enumerate(input.chunk(input.size(0), dim=0)):  # seq len만큼 수행함
            h_t_enc[0], c_t_enc[0] = self.enccell_list[0](input_mask[:, :input_dim] * input_t.squeeze(0), (
                input_mask[:, input_dim:input_dim + hidden_size] * h_t_enc[0],
                c_t_enc[0]))
            for ii in range(1, numlayers):
                h_t_enc[ii], c_t_enc[ii] = self.enccell_list[ii](mask_list[ii - 1][:, :hidden_size] * h_t_enc[ii - 1],
                                                                 (mask_list[ii - 1][:, hidden_size:2 * hidden_size] *
                                                                  h_t_enc[ii],
                                                                  c_t_enc[ii]))

            output = self.fc_enc(fc_mask * h_t_enc[-1])  # fully-connected layer

        for ii in range(numlayers):
            h_t_dec[ii] = h_t_enc[ii].clone()
            c_t_dec[ii] = c_t_enc[ii].clone()
            # print("output_enc",output)

        h_t_dec[0], c_t_dec[0] = self.deccell_list[0](input_mask2[:, :input_dim] * output, (
            input_mask2[:, input_dim:input_dim + hidden_size] * h_t_dec[0],
            c_t_dec[0]))

        for ii in range(1, numlayers):
            h_t_dec[ii], c_t_dec[ii] = self.deccell_list[ii](mask_list2[ii - 1][:, :hidden_size] * h_t_dec[ii - 1],
                                                             (mask_list2[ii - 1][:, hidden_size:2 * hidden_size] *
                                                              h_t_dec[ii],
                                                              c_t_dec[ii]))
        # output = self.batchnorm_dec(self.fcnorm_dec(self.dropout_fc_dec(h_t_dec[-1])))

        output = self.fc_dec(fc_mask2 * h_t_dec[-1])
        outputs += [output]

        # dec
        if ((self.training == True) or ((self.training == False) and (conditional == True))):
            for i, input_t in enumerate(input_r.chunk(input_r.size(0), dim=0)):
                if i != input_r.size(0) - 1:
                    h_t_dec[0], c_t_dec[0] = self.deccell_list[0](input_mask2[:, :input_dim] * input_t.squeeze(0), (
                        input_mask2[:, input_dim:input_dim + hidden_size] * h_t_dec[0], c_t_dec[0]))

                    for ii in range(1, numlayers):
                        h_t_dec[ii], c_t_dec[ii] = self.deccell_list[ii](
                            mask_list2[ii - 1][:, :hidden_size] * h_t_dec[ii - 1],
                            (mask_list2[ii - 1][:, hidden_size:2 * hidden_size] * h_t_dec[ii], c_t_dec[ii]))
                    # output = self.batchnorm_dec(self.fcnorm_dec(self.dropout_fc_dec(h_t_dec[-1])))
                    output = self.fc_dec(fc_mask2 * h_t_dec[-1])
                    outputs += [output]

        elif ((self.training == False) and (conditional == False)):
            for i in range(input_r.size(0)):
                if i != input_r.size(0) - 1:
                    h_t_dec[0], c_t_dec[0] = self.deccell_list[0](input_mask2[:, :input_dim] * output, (
                        input_mask2[:, input_dim:input_dim + hidden_size] * h_t_dec[0], c_t_dec[0]))

                    for ii in range(1, numlayers):
                        h_t_dec[ii], c_t_dec[ii] = self.deccell_list[ii](
                            mask_list2[ii - 1][:, :hidden_size] * h_t_dec[ii - 1],
                            (mask_list2[ii - 1][:, hidden_size:2 * hidden_size] * h_t_dec[ii], c_t_dec[ii]))

                    # output = self.batchnorm_dec(self.fcnorm_dec(self.dropout_fc_dec(h_t_dec[-1])))
                    output = self.fc_dec(fc_mask2 * h_t_dec[-1])
                    outputs += [output]

        outputs = torch.stack(outputs, 0)
        return outputs, h_t_enc, c_t_enc, h_t_dec, c_t_dec