# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from ..graphics import show_n_panels, load_thumbs


import numpy as np
import matplotlib.pyplot as plt


class ManualTracking:
    '''
    '''
    def __init__(self, cluster,
                 t0=0, n_panels=3, trail_length=10):

        self.t0 = t0
        self.n_panels = n_panels
        self.trail_length = trail_length
        self.cluster = cluster
        self.thumbs = load_thumbs(cluster, reset_ROI=True)
        self.figure, self.data_dict = show_n_panels(cluster, self.thumbs, t0,
                                                     self.n_panels,
                                                     self.trail_length)
        self.canvas = self.figure.canvas
        self.cid = self.canvas.mpl_connect('button_press_event', self)
        self.cid2 = self.canvas.mpl_connect('key_press_event', self)
        self.xs = []
        self.ys = []
        self.to_close = []
        self.backups = []

    def __call__(self, event):
        print(event.key)
        if not hasattr(event, 'button'):
            if event.key == 'ctrl+z':
                if len(self.backups):
                    self.cluster.trajs = self.backups[-1]
                    self.__update__()

            elif event.key in [' ', 'right', 'up']:
                self.forward()
            elif event.key in ['left', 'down']:
                self.backward()
            elif event.key in ['c', 'C', 'enter']:
                if len(self.to_close) == 2:
                    self.backups.append(self.cluster.trajs.copy())
                    close_gap(self.tracker,
                              self.to_close[0],
                              self.to_close[1])
                    self.__update__()
                else:
                    event.inaxes.set_title('Please choose only two trajectories')
                    self.__update__()
            elif event.key in ['x', 'X']:
                if len(self.to_close) == 1:
                    self.backups.append(self.cluster.trajs.copy())
                    cut_at_point(self.cluster,
                                 self.to_close[0], self.t0)
                    self.__update__()
                else:
                    event.inaxes.set_title('Please choose only two trajectories')
                    self.__update__()

        else:
            tb = plt.get_current_fig_manager().toolbar
            if tb.mode != '': return
            if event.button == 1:
                line = _closest_line(event)
                self.to_close.append(self.data_dict[line])
                event.inaxes.plot(line.get_data()[0],
                                  line.get_data()[1], 'ks-', mfc='None')
            elif event.button == 3:
                self.to_close.pop()
                event.inaxes.lines.pop()
            self.canvas.draw()

    def forward(self):
        self.t0 += 1
        self.__update__()

    def backward(self):

        if self.t0 > 0:
            self.t0 -= 1
        self.__update__()

    def __update__(self):

        self.figure, self.data_dict = show_n_panels(
            self.cluster,
            self.thumbs, self.t0,
            self.n_panels,
            trail_length=self.trail_length,
            fig=self.figure)
        self.to_close = []


def close_gap(trajs, label0, label1):

    new_labels = trajs.index.get_level_values('label').values
    new_labels[new_labels == label1] = label0
    trajs.relabel(new_labels)

def cut_at_point(trajs, label, t):

    new_labels = trajs.index.get_level_values('label').values
    times = trajs.index.get_level_values('t_stamp').values
    points = (new_labels == label) & (times >= t)
    new_labels[points] = new_labels.max() + 1
    trajs.relabel(new_labels)

def _closest_line(event):

    ax = event.inaxes
    min_dist = np.inf
    for line in ax.lines:
        xs, ys = line.get_data()
        dx = xs - event.xdata
        dy = ys - event.ydata
        dist = np.hypot(dy, dx)
        if min(dist) < min_dist:
            min_dist = min(dist)
            closest = line
    return closest