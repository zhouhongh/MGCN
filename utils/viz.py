#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2021/1/19 12:26
# @Author  : zhouhonghong
# @Email   : zhouhonghong@bupt.edu.cn
"""
    Functions to visualize human poses
    adapted from https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/viz.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# import h5py
import os
from mpl_toolkits.mplot3d import Axes3D
from utils import forward_kinematics as fk
from matplotlib.animation import FuncAnimation


class Ax3DPose(object):
    def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c", label=['GT', 'Pred']):
        """
        Create a 3d pose visualizer that can be updated with new poses.

        Args
          ax: 3d axis to plot the 3d pose on
          lcolor: String. Colour for the left part of the body
          rcolor: String. Colour for the right part of the body
        """

        # Start and endpoints of our representation
        self.I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1
        self.J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1
        # Left / right indicator
        self.LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
        self.ax = ax

        vals = np.zeros((32, 3))

        # Make connection matrix
        self.plots = []
        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            if i == 0:
                self.plots.append(
                    self.ax.plot(x, z, y, lw=2, linestyle='--', c=rcolor if self.LR[i] else lcolor, label=label[0]))
            else:
                self.plots.append(self.ax.plot(x, y, z, lw=2, linestyle='--', c=rcolor if self.LR[i] else lcolor))

        self.plots_pred = []
        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            if i == 0:
                self.plots_pred.append(self.ax.plot(x, y, z, lw=2, c=rcolor if self.LR[i] else lcolor, label=label[1]))
            else:
                self.plots_pred.append(self.ax.plot(x, y, z, lw=2, c=rcolor if self.LR[i] else lcolor))

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        # self.ax.set_axis_off()
        # self.ax.axes.get_xaxis().set_visible(False)
        # self.axes.get_yaxis().set_visible(False)
        self.ax.legend(loc='lower left')
        self.ax.view_init(120, -90)

    def update(self, gt_channels, pred_channels):
        """
        Update the plotted 3d pose.

        Args
          channels: 96-dim long np array. The pose to plot.
          lcolor: String. Colour for the left part of the body.
          rcolor: String. Colour for the right part of the body.
        Returns
          Nothing. Simply updates the axis with the new pose.
        """

        assert gt_channels.size == 96, "channels should have 96 entries, it has %d instead" % gt_channels.size
        gt_vals = np.reshape(gt_channels, (32, -1))
        lcolor = "#8e8e8e"
        rcolor = "#383838"
        for i in np.arange(len(self.I)):
            x = np.array([gt_vals[self.I[i], 0], gt_vals[self.J[i], 0]])
            y = np.array([gt_vals[self.I[i], 1], gt_vals[self.J[i], 1]])
            z = np.array([gt_vals[self.I[i], 2], gt_vals[self.J[i], 2]])
            self.plots[i][0].set_xdata(x)
            self.plots[i][0].set_ydata(y)
            self.plots[i][0].set_3d_properties(z)
            self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)
            # self.plots[i][0].set_alpha(0.5)

        assert pred_channels.size == 96, "channels should have 96 entries, it has %d instead" % pred_channels.size
        pred_vals = np.reshape(pred_channels, (32, -1))
        lcolor = "#9b59b6"
        rcolor = "#2ecc71"
        for i in np.arange(len(self.I)):
            x = np.array([pred_vals[self.I[i], 0], pred_vals[self.J[i], 0]])
            y = np.array([pred_vals[self.I[i], 1], pred_vals[self.J[i], 1]])
            z = np.array([pred_vals[self.I[i], 2], pred_vals[self.J[i], 2]])
            self.plots_pred[i][0].set_xdata(x)
            self.plots_pred[i][0].set_ydata(y)
            self.plots_pred[i][0].set_3d_properties(z)
            self.plots_pred[i][0].set_color(lcolor if self.LR[i] else rcolor)
            # self.plots_pred[i][0].set_alpha(0.7)

        r = 750
        xroot, yroot, zroot = gt_vals[0, 0], gt_vals[0, 1], gt_vals[0, 2]
        self.ax.set_xlim3d([-r + xroot, r + xroot])
        self.ax.set_zlim3d([-r + zroot, r + zroot])
        self.ax.set_ylim3d([-r + yroot, r + yroot])
        self.ax.set_aspect('equal')


def plot_predictions(expmap_gt, expmap_pred, fig, ax, f_title):
    # Load all the data
    parent, offset, rotInd, expmapInd = fk._some_variables()

    nframes_pred = expmap_pred.shape[0]

    # Compute 3d points for each frame
    xyz_gt = np.zeros((nframes_pred, 96))
    for i in range(nframes_pred):
        xyz_gt[i, :] = fk.fkl(expmap_gt[i, :], parent, offset, rotInd, expmapInd).reshape([96])
    xyz_pred = np.zeros((nframes_pred, 96))
    for i in range(nframes_pred):
        xyz_pred[i, :] = fk.fkl(expmap_pred[i, :], parent, offset, rotInd, expmapInd).reshape([96])

    # === Plot and animate ===
    ob = Ax3DPose(ax)
    # Plot the prediction
    for i in range(nframes_pred):

        ob.update(xyz_gt[i, :], xyz_pred[i, :])
        ax.set_title(f_title + ' frame:{:d}'.format(i + 1), loc="left")
        # plt.show(block=False)
        plt.savefig('/home/ubuntu/users/zhouhonghong/codes/Human_motion_prediction/MGCN_v2/test_savefig/{:d}.png'.format(i+1))
        fig.canvas.draw()
        # plt.pause(0.05)

## zhou
def plot_animation(predict, labels, filename):
    predict_plot = plot_h36m(predict, labels, filename)
    return predict_plot

class plot_h36m(object):

    def __init__(self, expmap_pred, expmap_gt, filename):
        # Load all the data
        parent, offset, rotInd, expmapInd = fk._some_variables()

        nframes_pred = expmap_pred.shape[0]

        # Compute 3d points for each frame
        xyz_gt = np.zeros((nframes_pred, 32, 3))
        for i in range(nframes_pred):
            xyz_gt[i, :, :] = fk.fkl(expmap_gt[i, :], parent, offset, rotInd, expmapInd).reshape([32, 3])
        xyz_pred = np.zeros((nframes_pred, 32,3))
        for i in range(nframes_pred):
            xyz_pred[i, :, :] = fk.fkl(expmap_pred[i, :], parent, offset, rotInd, expmapInd).reshape([32, 3])

        self.joint_xyz = xyz_gt
        self.nframes = xyz_gt.shape[0]
        self.joint_xyz_f = xyz_pred
        self.folder_dir = './h36m_long/'
        matplotlib.rc('axes', edgecolor=(1.0, 1.0, 1.0, 0.0))
        # set up the axes
        xmin = -750
        xmax = 750
        ymin = -750
        ymax = 750
        zmin = -750
        zmax = 750

        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax), projection='3d')
        self.ax.grid(False)
        self.ax.w_xaxis.set_pane_color((1.0, 1.0, 0.0, 0.0))
        self.ax.w_yaxis.set_pane_color((1.0, 1.0, 0.0, 0.0))
        self.ax.set_xticks([])
        self.ax.set_zticks([])
        self.ax.set_yticks([])

        self.chain = [np.array([0, 1, 2, 3, 4, 5]),
                     np.array([0, 6, 7, 8, 9, 10]),
                     np.array([0, 12, 13, 14, 15]),
                     np.array([13, 17, 18, 19, 22, 19, 21]),
                     np.array([13, 25, 26, 27, 30, 27, 29])]
        self.scats = []
        self.lns = []
        self.filename = filename

    def update(self, frame):
        for scat in self.scats:
            scat.remove()
        for ln in self.lns:
            self.ax.lines.pop(0)

        self.scats = []
        self.lns = []

        xdata = np.squeeze(self.joint_xyz[frame, :, 0])
        ydata = np.squeeze(self.joint_xyz[frame, :, 1])
        zdata = np.squeeze(self.joint_xyz[frame, :, 2])

        xdata_f = np.squeeze(self.joint_xyz_f[frame, :, 0])
        ydata_f = np.squeeze(self.joint_xyz_f[frame, :, 1])
        zdata_f = np.squeeze(self.joint_xyz_f[frame, :, 2])

        for i in range(len(self.chain)):
            self.lns.append(self.ax.plot3D(xdata_f[self.chain[i][:],], ydata_f[self.chain[i][:],], zdata_f[self.chain[i][:],], linewidth=2.0, color='#f94e3e')) # red: prediction
            self.lns.append(self.ax.plot3D(xdata[self.chain[i][:],], ydata[self.chain[i][:],], zdata[self.chain[i][:],], linewidth=2.0, color='#0780ea')) # blue: ground truth

    def plot(self):

        ani = FuncAnimation(self.fig, self.update, frames=self.nframes, interval=40, repeat=False)
        plt.title(self.filename, fontsize=16)
        ani.save(self.folder_dir + self.filename + '.gif', writer='pillow')
        plt.close()

