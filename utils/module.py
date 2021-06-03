#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2021/1/12 16:48
# @Author  : zhouhonghong
# @Email   : zhouhonghong@bupt.edu.cn
import torch.nn.functional as F
import torch.nn as nn
import torch

class AggregateJoint(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.mlps = nn.ModuleList()
        self.part_list = opt.s2_split
        for i in range(len(self.part_list)):
            input_dim = len(self.part_list[i])
            output_dim = 3
            hidden_dim = 16
            self.mlps.append(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(hidden_dim, output_dim),
                    nn.BatchNorm1d(output_dim),
                    nn.LeakyReLU()
                )
            )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x): # [B, N, F]
        b, n, f = x.shape
        x = x.permute(0, 2, 1).contiguous().view(-1,n) # [b*f,n]
        x_part = []
        for i in range(len(self.part_list)):
            x_part.append(self.mlps[i](x[:, self.part_list[i]])) # [b*f,3]
        x_part = torch.cat(x_part, dim=-1)  # [b*f,30]
        x_part = x_part.view(b,f,-1).permute(0,2,1).contiguous() # [b,30,f]
        return x_part


class AggregatePart(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.mlps = nn.ModuleList()
        self.part_list = opt.s3_split

        for i in range(len(self.part_list)):
            input_dim = len(self.part_list[i])
            output_dim = 3
            hidden_dim = 16
            self.mlps.append(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(hidden_dim, output_dim),
                    nn.BatchNorm1d(output_dim),
                    nn.LeakyReLU()
                )
            )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        b, n, f = x.shape
        x = x.permute(0, 2, 1).contiguous().view(-1, n)  # [b*f,n]
        x_body = []
        for i in range(len(self.part_list)):
            x_body.append(self.mlps[i](x[:, self.part_list[i]]))  # [b*f,3]
        x_body = torch.cat(x_body, dim=-1)  # [b*f,15]
        x_body = x_body.view(b, f, -1).permute(0, 2, 1).contiguous()  # [b,15,f]
        return x_body

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, do_prob1, do_prob2, node_n):

        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.act_f1 = nn.LeakyReLU(negative_slope=0.01)
        self.act_f2 = nn.LeakyReLU(negative_slope=0.01)
        self.act_f3 = nn.LeakyReLU(negative_slope=0.01)
        self.bn = nn.BatchNorm1d(node_n * output_dim)
        self.drop1 = nn.Dropout(p=do_prob1)
        self.drop2 = nn.Dropout(p=do_prob2)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        """
        :param inputs: [b,n,f]
        :return:[b,n]
        """
        x = self.fc1(inputs)
        x = self.act_f1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.act_f2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        b, n, f = x.shape
        x = self.bn(x.view(b, -1)).view(b, n, f)
        x = self.act_f3(x)
        return x

class CrossScale_S1_to_S2(nn.Module):
    def __init__(self, feature_dim, node_s1, node_s2):
        super().__init__()
        self.h_s1 = MLP(feature_dim, 256, 512, 0.2, 0.5, node_n=node_s1)
        self.h_s2 = MLP(feature_dim, 256, 512, 0.2, 0.5, node_n=node_s2)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x_s1, x_s2):
        s1_att = self.h_s1(x_s1)
        s2_att = self.h_s2(x_s2)
        Att = self.softmax(torch.matmul(s2_att, s1_att.permute(0,2,1)))
        x_s2_cross = torch.einsum('bnw, bwf->bnf', (Att, x_s1))
        return x_s2_cross

class CrossScale_S2_to_S1(nn.Module):
    def __init__(self, feature_dim, node_s1, node_s2):
        super().__init__()
        self.h_s1 = MLP(feature_dim, 256, 512, 0.2, 0.5, node_n=node_s1)
        self.h_s2 = MLP(feature_dim, 256, 512, 0.2, 0.5, node_n=node_s2)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x_s2, x_s1):
        s1_att = self.h_s1(x_s1)
        s2_att = self.h_s2(x_s2)
        Att = self.softmax(torch.matmul(s1_att, s2_att.permute(0,2,1)))
        x_s1_cross = torch.einsum('bnw, bwf->bnf', (Att, x_s2))
        return x_s1_cross

class CrossScale_S2_to_S3(nn.Module):
    def __init__(self, feature_dim, node_s2, node_s3):
        super().__init__()
        self.h_s3 = MLP(feature_dim, 256, 512, 0.2, 0.5, node_n=node_s3)
        self.h_s2 = MLP(feature_dim, 256, 512, 0.2, 0.5, node_n=node_s2)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x_s2, x_s3):
        s3_att = self.h_s3(x_s3)
        s2_att = self.h_s2(x_s2)
        Att = self.softmax(torch.matmul(s3_att, s2_att.permute(0,2,1)))
        x_s3_cross = torch.einsum('bnw, bwf->bnf', (Att, x_s2))
        return x_s3_cross

class CrossScale_S3_to_S2(nn.Module):
    def __init__(self, feature_dim, node_s2, node_s3):
        super().__init__()
        self.h_s3 = MLP(feature_dim, 256, 512, 0.2, 0.5, node_n=node_s3)
        self.h_s2 = MLP(feature_dim, 256, 512, 0.2, 0.5, node_n=node_s2)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x_s3, x_s2):
        s3_att = self.h_s3(x_s3)
        s2_att = self.h_s2(x_s2)
        Att = self.softmax(torch.matmul(s2_att, s3_att.permute(0,2,1)))
        x_s3_cross = torch.einsum('bnw, bwf->bnf', (Att, x_s3))
        return x_s3_cross

class SplitPart(nn.Module):
    def __init__(self, opt):
        super(SplitPart, self).__init__()
        self.mlps = nn.ModuleList()
        self.part_list = opt.s2_split
        for i in range(len(self.part_list)):
            input_dim = 3
            output_dim = len(self.part_list[i])
            hidden_dim = 16
            self.mlps.append(
                nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.LeakyReLU()
                )
            )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        b, n, f = x.shape # n = 30
        x = x.permute(0, 2, 1).contiguous().view(-1, n)  # [b*f,n]
        x_joint = []
        for i in range(len(self.part_list)):
            x_joint.append(self.mlps[i](x[:, 3*i:3*i+3])) # [b*f,len(part_list[i])]
        x_joint = torch.cat(x_joint, dim=-1)  # [b*f,45]
        x_joint = x_joint.view(b, f, -1).permute(0, 2, 1).contiguous()  # [b,45,f]
        return x_joint

class SplitBody(nn.Module):
    def __init__(self, opt):
        super(SplitBody, self).__init__()
        self.mlps = nn.ModuleList()
        self.part_list = opt.s3_split
        for i in range(len(self.part_list)):
            input_dim = 3
            output_dim = len(self.part_list[i])
            hidden_dim = 16
            self.mlps.append(
                nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.LeakyReLU()
                )
            )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        b, n, f = x.shape # n = 15
        x = x.permute(0, 2, 1).contiguous().view(-1, n)  # [b*f,n]
        x_joint = []
        for i in range(len(self.part_list)):
            x_joint.append(self.mlps[i](x[:, 3*i:3*i+3])) # [b*f,len(part_list[i])]
        x_joint = torch.cat(x_joint, dim=-1)  # [b*f,45]
        x_joint = x_joint.view(b, f, -1).permute(0, 2, 1).contiguous()  # [b,45,f]
        return x_joint
