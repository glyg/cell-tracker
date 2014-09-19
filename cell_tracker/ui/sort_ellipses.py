# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from ..graphics import show_4panel_ellipses

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import logging
log = logging.getLogger(__name__)

from sktracker.trajectories import draw
from ..graphics import show_ellipses


'''
In this module, we set up an interactive tool to remove false positives from detected ellipses
'''


class EllipsisPicker:
    '''
    '''
    def __init__(self, cluster, sizes, coords=['x', 'y', 'z']):
        '''
        Creates a matplotlib graph with added interactions to select and suppres
        fitted ellipses

        Parameters:
        -----------

        cluster: a :class:`cell_tracker.Cluster` instance
        sizes

        '''
        self.cluster = cluster
        self.sizes = sizes
        self.coords = coords

        self.axes = None
        self.ax_3d = None

        self.labels = list(cluster.trajs.labels)
        self.current_label = self.labels[0]
        self.traj_data = set()
        self.bad_ellipses = []#set()
        self.bad_lines = []
        self.collected = []
        self.backups = []
        self.overlays = []
        self.__init_fig()


    def __init_fig(self):
        self.figure, self.axes = plt.subplots(2, 2,
                                 sharex='col',
                                 sharey='row', figsize=(12, 12))

        for ax in self.axes.ravel():
            ax.set_aspect('equal')
        self.axes[1, 1].axis('off')
        self.ax_3d = self.figure.add_subplot(224, projection='3d')

        self.axes, self.ax_3d = draw.show_4panels(self.cluster.trajs, self.current_label,
                                                  axes=self.axes, ax_3d=self.ax_3d,
                                                  coords=self.coords)
        for ax in self.axes.ravel():
            self.traj_data.update(set(ax.lines))
        self.traj_data.update(set(self.ax_3d.lines))
        self.figure = self.axes[0, 0].get_figure()
        self.canvas = self.figure.canvas
        self.cid = self.canvas.mpl_connect('button_press_event', self)
        self.cid2 = self.canvas.mpl_connect('key_press_event', self)
        self.show_pickable_ellipses()


    def show_pickable_ellipses(self):
        ellipses_kwargs = {'c':'k', 'lw': 1.5, 'alpha':0.4}

        self.ellipses_lines = []
        for size in self.sizes:
            self.axes, self.ax_3d, lines_df = show_ellipses(self.cluster,
                                                            self.current_label,
                                                            size,
                                                            cutoffs=None,
                                                            coords=self.coords,
                                                            axes=self.axes, ax_3d=self.ax_3d,
                                                            return_lines=True,
                                                            show_centers=False,
                                                            **ellipses_kwargs)
            self.ellipses_lines.append(lines_df)
        self.ellipses_lines = pd.concat(self.ellipses_lines).sortlevel()
        self.axes[0,0].set_title(self.current_label)
        plt.draw()

    def __call__(self, event):

        self.curent_event = event
        if event.inaxes == self.ax_3d: return
        if not hasattr(event, 'button'):
            log.info(event.key)
            print(event.key)
            # if event.key == 'ctrl+z':
            #     if len(self.backups):
            #         self.cluster.trajs = self.backups[-1]
            #         self.__update__()

            if event.key in ['pageup', 'ctr+right', 'alt+f']:#in [' ', 'right', 'up']:
                print('forward + {}'.format(event.key))
                self.forward()
            elif event.key in ['pagedown', 'ctr+left', 'alt+b']:# in ['left', 'down']:
                print('backward + {}'.format(event.key))
                self.backward()
            elif event.key in ['x', 'X']:
                self.remove_bad(event)
        else:
            tb = plt.get_current_fig_manager().toolbar
            if tb.mode != '': return
            if event.button == 1:
                line = self._closest_line(event)
                self.mark_bad(line, event)
                #plt.draw()
            elif event.button == 3:
                if len(self.bad_ellipses):
                    self.unmark_bad(event)
            self.canvas.draw()

    def mark_bad(self, close_line, event):
        ### Register the data point
        tls = tuple(self.ellipses_lines.loc[event.inaxes, close_line].values) # time, label, size
        if tls in self.bad_ellipses:
            return
        self.bad_ellipses.append(tls)
        where = self._reverse_search(tls)
        self.bad_lines.append(where)
        for ax, line in where.index:
            if ax == self.ax_3d:
                ### Plots overlay
                overlay = ax.plot(line.get_data()[0],
                                  line.get_data()[1],
                                  'k-', lw=2, alpha=0.8)
            else:
                overlay = ax.plot(line.get_data()[0],
                                  line.get_data()[1],
                                  'k-', lw=2, alpha=0.8)
            self.overlays.append((ax, overlay[0]))

    def iter_axes(self):
        axes = (self.axes[0, 0], self.axes[0, 0], self.axes[0, 0], self.ax_3d)
        for ax in axes:
            yield ax

    def unmark_bad(self, event):

        self.bad_ellipses.pop()
        self.bad_lines.pop()
        ### Plots overlay
        for i in range(4):
            ax, overlay =  self.overlays.pop()
            try:
                ax.lines.remove(overlay)
            except ValueError:
                pass

    def remove_bad(self, event):
        # if not len(self.bad_ellipses):
        #     return
        # for s, t, l in self.bad_ellipses:
        #     self.cluster.ellipses.loc[s, t, l]['good'] = 100
        if not len(self.bad_lines):
            return
        for where in self.bad_lines:
            for ax, line in where.index:
                try:
                    ax.lines.remove(line)
                except ValueError:
                    pass
        for ax, overlay in self.overlays:
            try:
                ax.lines.remove(overlay)
            except ValueError:
                continue
        self.bad_lines = []
        self.overlays = []
        for tls in self.bad_ellipses:
            self.cluster.ellipses.loc[tls, 'good'] = -1

        plt.draw()

    def forward(self):

        if self.current_label != self.labels[-1]:
            self.current_label = self.labels[self.labels.index(self.current_label)+1]
            self.__update__()
        else:
            self.axes[0, 0].set_title('No more label before')

    def backward(self):

        if self.current_label != self.labels[0]:
            self.current_label = self.labels[self.labels.index(self.current_label)-1]
            self.__update__()
        else:
            self.axes[0, 0].set_title('No more label after')

    def __update__(self):
        print('update')
        for ax in self.axes.ravel():
            ax.cla()
        self.ax_3d.cla()
        self.traj_data = set()
        self.axes, self.ax_3d = draw.show_4panels(self.cluster.trajs, self.current_label,
                                                  axes=self.axes, ax_3d=self.ax_3d,
                                                  coords=self.coords)
        for ax in self.axes.ravel():
            self.traj_data.update(set(ax.lines))
        self.show_pickable_ellipses()

    def _closest_line(self, event):

        ax = event.inaxes
        min_dist = np.inf
        for line in ax.lines:
            if line in self.traj_data:
                continue
            elif line in self.overlays:
                continue
            xs, ys = line.get_data()
            dx = xs - event.xdata
            dy = ys - event.ydata
            dist = np.hypot(dy, dx)
            if min(dist) < min_dist:
                min_dist = min(dist)
                closest = line
        return closest

    def _reverse_search(self, tls):

        t, l, s = tls
        ts = self.ellipses_lines['t_stamp'] == t
        ls = self.ellipses_lines['label'] == l
        ss = self.ellipses_lines['size'] == s
        return self.ellipses_lines[ts & ls & ss]