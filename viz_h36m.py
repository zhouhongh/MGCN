#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2021/5/31 17:20
# @Author  : zhouhonghong
# @Email   : zhouhonghong@bupt.edu.cn

"""
use the checkpoint to visualize the motions predicted on the CMU mocap dataset
"""
from __future__ import print_function, absolute_import, division

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import time
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional
import numpy as np
from progress.bar import Bar
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import loss_funcs, utils as utils
from utils.opt import Options
from utils.h36motion import H36motion
import utils.model as nnmodel
import utils.data_utils as data_utils
from tqdm import tqdm
import imageio



class Ax3DPose(object):
    def __init__(self, ax, p1_lcolor="#3498db", p1_rcolor="#e74c3c",label=['GT', 'Pred']):
        """
        Create a 3d pose visualizer that can be updated with new poses.

        Args
          ax: 3d axis to plot the 3d pose on
          lcolor: String. Colour for the left part of the body
          rcolor: String. Colour for the right part of the body
        """

        # Start and endpoints of two persons
        self.I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1
        self.J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1
        """
        Left / right  indicator:
        pre p1 left: 0
        pre p1 right: 1
        gt: 2
        """
        self.color_ind = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int)
        switch = {
            0: p1_lcolor,
            1: p1_rcolor,
            2: "#BEBEBE"
        }
        self.ax = ax

        vals = np.zeros((38, 3))

        # Make connection matrix
        self.plots = []
        color = switch[2]
        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])

            if i == 0:
                self.plots.append(
                    self.ax.plot(x, z, y, lw=2, linestyle='--', c=color, label=label[0]))
            else:
                self.plots.append(self.ax.plot(x, y, z, lw=2, linestyle='--', c=color))

        self.plots_pred = []
        for i in np.arange(len(self.I)):
            color = switch[self.color_ind[i]]
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            if i == 0:
                self.plots_pred.append(self.ax.plot(x, y, z, lw=2, c=color, label=label[1]))
            else:
                self.plots_pred.append(self.ax.plot(x, y, z, lw=2, c=color))


        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.ax.set_axis_off()
        # self.ax.axes.get_xaxis().set_visible(False)
        # self.axes.get_yaxis().set_visible(False)
        self.ax.legend(loc='lower left')
        self.ax.view_init(120, -90)

    def update(self, gt_vals, pred_vals):
        """
        Update the plotted 3d pose.

        Args
          lcolor: String. Colour for the left part of the body.
          rcolor: String. Colour for the right part of the body.
        Returns
          Nothing. Simply updates the axis with the new pose.
        """
        switch = {
            0: "#3498db",
            1: "#e74c3c",
            2: "#BEBEBE"
        }
        color = switch[2]
        for i in np.arange(len(self.I)):
            x = np.array([gt_vals[self.I[i], 0], gt_vals[self.J[i], 0]])
            y = np.array([gt_vals[self.I[i], 1], gt_vals[self.J[i], 1]])
            z = np.array([gt_vals[self.I[i], 2], gt_vals[self.J[i], 2]])
            self.plots[i][0].set_xdata(x)
            self.plots[i][0].set_ydata(y)
            self.plots[i][0].set_3d_properties(z)
            self.plots[i][0].set_color(color)
            # self.plots_pred[i][0].set_alpha(0.7)

        for i in np.arange(len(self.I)):
            color = switch[self.color_ind[i]]
            x = np.array([pred_vals[self.I[i], 0], pred_vals[self.J[i], 0]])
            y = np.array([pred_vals[self.I[i], 1], pred_vals[self.J[i], 1]])
            z = np.array([pred_vals[self.I[i], 2], pred_vals[self.J[i], 2]])
            self.plots_pred[i][0].set_xdata(x)
            self.plots_pred[i][0].set_ydata(y)
            self.plots_pred[i][0].set_3d_properties(z)
            self.plots_pred[i][0].set_color(color)
        r = 750
        xroot, yroot, zroot = gt_vals[0, 0], gt_vals[0, 1], gt_vals[0, 2]
        self.ax.set_xlim3d([-r + xroot, r + xroot])
        self.ax.set_zlim3d([-r + zroot, r + zroot])
        self.ax.set_ylim3d([-r + yroot, r + yroot])
        self.ax.set_aspect('equal')


def plot_predictions(gt_3d, pred_3d, ax, f_title, imgs_path):
    # Load all the data

    nframes_pred = gt_3d.shape[0]

    # === Plot and animate ===
    ob = Ax3DPose(ax)
    # Plot the prediction
    for i in range(nframes_pred):
        ob.update(gt_3d[i, :], pred_3d[i, :])
        f_title_new = f_title + 'frame:{}'.format(i + 1)
        jpg_name = '%03d'%(i+1)+'.jpg'
        ax.set_title(f_title_new, loc="left")
        # plt.show(block=False)
        seq_folder = os.path.join(imgs_path, f_title)
        if not os.path.exists(seq_folder):
            os.makedirs(seq_folder)
        plt.savefig(os.path.join(seq_folder, jpg_name))
        # fig.canvas.draw()

def save_gif(imgs_root, gif_root):
    if not os.path.exists(gif_root):
        os.makedirs(gif_root)
    for img_folder in tqdm(os.listdir(imgs_root)):
        img_path = os.path.join(imgs_root, img_folder)
        images = []
        imgs = sorted((fn for fn in os.listdir(img_path) if fn.endswith('.jpg')))
        for img in imgs:
            images.append(imageio.imread(os.path.join(img_path, img)))
        imageio.mimsave(os.path.join(gif_root, img_folder.strip(',') + '.gif'), images, duration=0.2)

def test(data_loader, model, input_n=20, output_n=50, is_cuda=False, dim_used=[], dct_n=20):
    targ_all_3d = None
    pred_all_3d = None
    for i, (inputs, targets, all_seq) in enumerate(data_loader):
        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            all_seq = Variable(all_seq.cuda(async=True)).float()

        outputs = model(inputs)
        n = outputs.shape[0]
        # outputs = outputs.view(n, -1)
        # targets = targets.view(n, -1)

        # loss = loss_funcs.sen_loss(outputs, all_seq, dim_used)

        n, seq_len, dim_full_len = all_seq.data.shape
        dim_used_len = len(dim_used)

        # inverse dct transformation
        _, idct_m = data_utils.get_dct_matrix(seq_len)
        idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
        outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
        outputs_exp = torch.matmul(idct_m[:, :dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len,
                                                                                                   seq_len).transpose(1,
                                                                                                                      2)

        pred_expmap = all_seq.clone()
        dim_used = np.array(dim_used)
        pred_expmap[:, :, dim_used] = outputs_exp
        pred_expmap = pred_expmap.contiguous().view(-1, dim_full_len)
        targ_expmap = all_seq.clone().contiguous().view(-1, dim_full_len)

        pred_expmap[:, 0:6] = 0
        targ_expmap[:, 0:6] = 0
        pred_expmap = pred_expmap.view(-1, 3)
        targ_expmap = targ_expmap.view(-1, 3)

        # get 3d coordinates
        targ_p3d = data_utils.expmap2xyz_torch(targ_expmap.view(-1, dim_full_len)).view(n, input_n+output_n, -1, 3)
        pred_p3d = data_utils.expmap2xyz_torch(pred_expmap.view(-1, dim_full_len)).view(n, input_n+output_n, -1, 3)
        if i==0:
            targ_all_3d = targ_p3d
            pred_all_3d = pred_p3d
        else:
            targ_all_3d = torch.cat((targ_all_3d, targ_p3d))
            pred_all_3d = torch.cat((pred_all_3d, pred_p3d))

    return targ_all_3d, pred_all_3d

# command
# python demo.py --input_n 10 --output_n 10 --dct_n 20 --data_dir /mnt/DataDrive164/zhouhonghong/h36m
def main(opt,img_path,model_path_short, model_path_long):
    is_cuda = torch.cuda.is_available()

    # create model
    print(">>> creating model")
    input_n = opt.input_n
    output_n = opt.output_n
    dct_n = opt.dct_n
    sample_rate = opt.sample_rate

    model = nnmodel.MGCN(opt)

    if is_cuda:
        model.cuda()

    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    opt.is_load = True
    if opt.is_load:
        if dct_n == 20:
            model_path_len = model_path_short
        else:
            model_path_len = model_path_long
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        if is_cuda:
            ckpt = torch.load(model_path_len)
        else:
            ckpt = torch.load(model_path_len, map_location='cpu')
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))

    # data loading
    train_dataset = H36motion(path_to_data=opt.data_dir, actions=opt.actions, input_n=input_n, output_n=output_n,
                              split=0, sample_rate=sample_rate, dct_n=dct_n, test_mode=opt.test_mode)
    data_std = train_dataset.data_std
    data_mean = train_dataset.data_mean

    acts = data_utils.define_actions(opt.actions)
    test_data = dict()
    for act in acts:
        test_dataset = H36motion(path_to_data=opt.data_dir, actions=act, input_n=input_n, output_n=output_n, split=1,
                                 sample_rate=sample_rate, data_mean=data_mean, data_std=data_std, dct_n=dct_n)
        test_data[act] = DataLoader(
            dataset=test_dataset,
            batch_size=opt.test_batch,
            shuffle=False,
            num_workers=opt.job,
            pin_memory=True)
    print(">>> data loaded !")
    print(">>> train data {}".format(train_dataset.__len__()))

    ax = plt.gca(projection='3d')
    for act in acts:
        # [8,10,38,3]
        targ_3d, pred_3d = test(test_data[act],model,input_n=input_n,output_n=output_n,is_cuda=is_cuda,dim_used=train_dataset.dim_used,dct_n=dct_n)
        for k in tqdm(range(targ_3d.shape[0])):
            plt.cla()
            figure_title = "act:{},seq:{},".format(act, k + 1)
            plot_predictions(targ_3d[k,:,:,:].detach().cpu().numpy(), pred_3d[k,:,:,:].detach().cpu().numpy(), ax, figure_title, img_path)


if __name__ == "__main__":
    option = Options().parse()
    main(option,
         img_path="/mnt/DataDrive164/zhouhonghong/outputs/MGCN/h36m_long/imgs",
         model_path_short='checkpoint/h36m_short/ckpt_main_in10_out10_dctn20_best.pth.tar',
         model_path_long='checkpoint/h36m_long/ckpt_main_in10_out25_dctn35_best.pth.tar')
    save_gif(imgs_root="/mnt/DataDrive164/zhouhonghong/outputs/MGCN/h36m_long/imgs",
             gif_root="/mnt/DataDrive164/zhouhonghong/outputs/MGCN/h36m_long/gifs",)

