#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2021/1/12 11:42
# @Author  : zhouhonghong
# @Email   : zhouhonghong@bupt.edu.cn

from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
import utils.module as Module


class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, hidden_feature, p_dropout, num_stage=1, node_n=48):
        """
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage
        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

    def forward(self, x):

        for i in range(self.num_stage):
            x = self.gcbs[i](x)

        return x

class MGCN(nn.Module):
    def __init__(self, opt):
        super(MGCN, self).__init__()
        input_feature = opt.dct_n
        hidden_feature = opt.linear_size
        p_dropout = opt.dropout
        num_stage = opt.num_stage
        self.num_modules = opt.num_modules
        node_s1 = opt.node_s1
        node_s2 = opt.node_s2
        node_s3 = opt.node_s3
        self.decoder = opt.decoder

        self.s2 = Module.AggregateJoint(opt)
        self.s3 = Module.AggregatePart(opt)
        self.split_s2 = Module.SplitPart(opt)
        self.split_s3 = Module.SplitBody(opt)

        self.gc1_s1 = GraphConvolution(input_feature, hidden_feature, node_n=node_s1)
        self.bn1_s1 = nn.BatchNorm1d(node_s1 * hidden_feature)
        self.gc1_s2 = GraphConvolution(input_feature, hidden_feature, node_n=node_s2)
        self.bn1_s2 = nn.BatchNorm1d(node_s2 * hidden_feature)
        self.gc1_s3 = GraphConvolution(input_feature, hidden_feature, node_n=node_s3)
        self.bn1_s3 = nn.BatchNorm1d(node_s3 * hidden_feature)
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

        self.gc_dec_s1 = GraphConvolution(hidden_feature, input_feature, node_n=node_s1)
        self.gc_dec_s2 = GraphConvolution(hidden_feature, hidden_feature, node_n=node_s1)
        self.gc_dec_s3 = GraphConvolution(hidden_feature, hidden_feature, node_n=node_s1)

        self.encoder_modules = nn.ModuleList()
        for i in range(self.num_modules):
            self.encoder_modules.append(
                self.multiscale_module(hidden_feature,p_dropout,num_stage,node_s1,node_s2,node_s3)
            )
        # for cross scale
        self.lambda_12 = 0.5
        self.lambda_32 = 0.5
        self.lambda_21 = 1.0
        self.lambda_23 = 1.0

    def multiscale_module(self, hidden_feature, p_dropout, num_stage, node_s1, node_s2, node_s3):

        return nn.ModuleList([
                GCN(hidden_feature, p_dropout, num_stage, node_s1),
                GCN(hidden_feature, p_dropout, num_stage, node_s2),
                GCN(hidden_feature, p_dropout, num_stage, node_s3),
                Module.CrossScale_S2_to_S1(hidden_feature, node_s1, node_s2),
                Module.CrossScale_S1_to_S2(hidden_feature, node_s1, node_s2),
                Module.CrossScale_S3_to_S2(hidden_feature, node_s2, node_s3),
                Module.CrossScale_S2_to_S3(hidden_feature, node_s2, node_s3)
            ])


    def gc1(self, x_s1, x_s2, x_s3):
        y_s1 = self.gc1_s1(x_s1)
        y_s2 = self.gc1_s2(x_s2)
        y_s3 = self.gc1_s3(x_s3)
        b, n, f = y_s1.shape
        y_s1 = self.bn1_s1(y_s1.view(b, -1)).view(b, n, f)
        b, n, f = y_s2.shape
        y_s2 = self.bn1_s2(y_s2.view(b, -1)).view(b, n, f)
        b, n, f = y_s3.shape
        y_s3 = self.bn1_s3(y_s3.view(b, -1)).view(b, n, f)
        y_s1 = self.act_f(y_s1)
        y_s2 = self.act_f(y_s2)
        y_s3 = self.act_f(y_s3)
        y_s1 = self.do(y_s1)
        y_s2 = self.do(y_s2)
        y_s3 = self.do(y_s3)
        return y_s1, y_s2, y_s3

    def forward(self, x):
        x_s1 = x
        x_s2 = self.s2(x)
        x_s3 = self.s3(x)
        y_s1, y_s2, y_s3 = self.gc1(x_s1, x_s2, x_s3)
        for i in range(self.num_modules):
            input_y1 = y_s1
            input_y2 = y_s2
            input_y3 = y_s3
            y1 = self.encoder_modules[i][0](y_s1)
            y2 = self.encoder_modules[i][1](y_s2)
            y3 = self.encoder_modules[i][2](y_s3)
            y_s1 = y1 + self.lambda_21 * self.encoder_modules[i][3](y2, y1)
            y_s2 = y2 + self.lambda_12 * self.encoder_modules[i][4](y1, y2) + self.lambda_32 * self.encoder_modules[i][
                5](y3, y2)
            y_s3 = y3 + self.lambda_23 * self.encoder_modules[i][6](y2, y3)
            y_s1 = y_s1 + input_y1
            y_s2 = y_s2 + input_y2
            y_s3 = y_s3 + input_y3
        # decode from s3 to s1
        y_s21 = self.split_s2(y_s2)
        y_s31 = self.split_s3(y_s3)

        # sequential decoder
        if self.decoder == 'sequential':
            y_s3_dec = self.gc_dec_s3(y_s31)
            y_s21 = y_s21 + y_s3_dec
            y_s2_dec = self.gc_dec_s2(y_s21)
            y_s1 = y_s1 + y_s2_dec
            y_s1_dec = self.gc_dec_s1(y_s1)

        # parallel decoder
        elif self.decoder == 'parallel':
            y_s3_dec = self.gc_dec_s3(y_s31)
            y_s2_dec = self.gc_dec_s2(y_s21)
            y_s1_dec = self.gc_dec_s1(y_s1)
            y_s1_dec = y_s3_dec + y_s2_dec + y_s1_dec
            y_s1_dec = self.gc_dec_final(y_s1_dec)

        y = y_s1_dec + x
        return y