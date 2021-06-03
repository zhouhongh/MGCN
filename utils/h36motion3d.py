#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2021/1/21 18:00
# @Author  : zhouhonghong
# @Email   : zhouhonghong@bupt.edu.cn
"""
adapted from https://github.com/wei-mao-2019/LearnTrajDep
"""
from torch.utils.data import Dataset
import numpy as np
from utils import data_utils


class H36motion3D(Dataset):

    def __init__(self, path_to_data, actions, input_n=20, output_n=10, dct_used=15, split=0, sample_rate=2,
                 test_mode=False):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = path_to_data
        self.split = split
        self.dct_used = dct_used

        subs = np.array([[1, 6, 7, 8, 9], [5], [11]])
        acts = data_utils.define_actions(actions)

        # subs = np.array([[1], [5], [11]])
        # acts = ['walking']

        subjs = subs[split]
        all_seqs, dim_ignore, dim_used = data_utils.load_data_3d(path_to_data, subjs, acts, sample_rate,
                                                                 input_n + output_n)
        self.all_seqs = all_seqs
        self.dim_used = dim_used
        all_seqs = all_seqs[:, :, dim_used]
        all_seqs = all_seqs.transpose(0, 2, 1)
        all_seqs = all_seqs.reshape(-1, input_n + output_n)
        all_seqs = all_seqs.transpose()

        dct_m_in, _ = data_utils.get_dct_matrix(input_n + output_n)
        dct_m_out, _ = data_utils.get_dct_matrix(input_n + output_n)
        pad_idx = np.repeat([input_n - 1], output_n)
        i_idx = np.append(np.arange(0, input_n), pad_idx)
        input_dct_seq = np.matmul(dct_m_in[0:dct_used, :], all_seqs[i_idx, :])
        input_dct_seq = input_dct_seq.transpose().reshape([-1, len(dim_used), dct_used])
        # input_dct_seq = input_dct_seq.reshape(-1, len(dim_used) * dct_used)

        output_dct_seq = np.matmul(dct_m_out[0:dct_used, :], all_seqs)
        output_dct_seq = output_dct_seq.transpose().reshape([-1, len(dim_used), dct_used])
        # output_dct_seq = output_dct_seq.reshape(-1, len(dim_used) * dct_used)

        self.input_dct_seq = input_dct_seq
        self.output_dct_seq = output_dct_seq
        if split == 0 and test_mode:
            keep_rate = 0.1
            data_len = self.input_dct_seq.shape[0]
            keep_id = np.linspace(0, data_len-1, int(keep_rate*data_len), dtype=int)
            self.input_dct_seq = self.input_dct_seq[keep_id, :]
            self.output_dct_seq = self.output_dct_seq[keep_id, :]
            self.all_seqs = self.all_seqs[keep_id, :]

    def __len__(self):
        return np.shape(self.input_dct_seq)[0]

    def __getitem__(self, item):
        return self.input_dct_seq[item], self.output_dct_seq[item], self.all_seqs[item]
