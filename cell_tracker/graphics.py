# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import warnings
import itertools

import logging
log = logging.getLogger(__name__)


#draw a vector
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d, Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import center_of_mass
from sktracker.trajectories import draw

from .analysis import Ellipses
from .objects import build_iterator

def _scatter_single_segment(cluster, label, sizes, axes=None):

    symbols = 'soDd'
    symbols = {size: symbol for size, symbol in zip(sizes, itertools.cycle(symbols))}

    if axes is None:
        fig, axes = plt.subplots(2, 2, sharex='col',
                                sharey='row', figsize=(12, 12))

    for size in sizes:
        ellipsis_df = cluster.ellipses.xs(size, level='size').xs(label, level='label').dropna()
        if 'color' in ellipsis_df.columns:
            color = ellipsis_df['color']
        else:
            color = 'k'

        if ellipsis_df.empty:
            continue
        gof = ellipsis_df['gof'].astype(np.float)
        gof_size = 50 * (gof - gof.min()) / gof.ptp()

        ellipticities = ellipsis_df['ellipticity'].astype(np.float)
        radius = ellipsis_df['radius'].astype(np.float)
        rad_size = 50 * (np.log(radius) - np.log(radius.min())) / np.log(radius.max())
        dtheta = ellipsis_df['dtheta'].astype(np.float) * 180 / np.pi

        axes[0, 0].scatter(gof, ellipticities,
                           c=color, marker=symbols[size],
                           alpha=0.5, s=rad_size)
        #axes[0, 0].plot([gof.min(), cutoffs['gof']
        axes[1, 0].scatter(gof, radius,
                           c=color, marker=symbols[size],
                           s= 50 / ellipticities , alpha=0.5)
        axes[0, 1].scatter(np.abs(dtheta), ellipticities,
                           c=color, marker=symbols[size],
                           alpha=0.5, s=gof_size)
        axes[1, 1].scatter(np.abs(dtheta), radius,
                           c=color, marker=symbols[size],
                           alpha=0.5, s=gof_size)



def show_ellipses_clusters(cluster, sizes, cutoffs, axes=None):
    if axes is None:
        fig, axes = plt.subplots(2, 2, sharex='col',
                                 sharey='row', figsize=(12, 12))

    color = cluster.trajs.get_colors()
    for label in cluster.trajs.labels:
        _scatter_single_segment(cluster, label, sizes, axes=axes)
    ### goodness of fit vs ellipticity

    ax = axes[0, 0]

    gof_cut = cutoffs['min_gof']
    ell_cut_max = cutoffs['max_ellipticity']

    ax.axvline(gof_cut, c='r',
                    lw=2, alpha=0.5)

    ax.axhline(ell_cut_max, c='k',
                    lw=2, alpha=0.5)


    ax.set_ylabel('Ellipticity')
    ax.set_title('Size: Avg radius', fontsize=12)
    ax.set_xlim(-1, 4)
    ax.set_ylim(0, 5)


    ### goodness of fit vs radius
    ax = axes[1, 0]

    rad_cut_min = cutoffs['min_radius']
    rad_cut_max = cutoffs['max_radius']

    ax.axvline(gof_cut, c='k',
                    lw=2, alpha=0.5)
    ax.axhline(rad_cut_min, c='k',
                    lw=2, alpha=0.5)
    ax.axhline(rad_cut_max, c='k',
                    lw=2, alpha=0.5)
    ax.set_xlabel('Goodness of fit')
    ax.set_ylabel('Average radius')
    ax.set_title('Size : 1/Ellipticity', fontsize=12)
    ax.set_yscale('log')
    axes[1, 1].set_yscale('log')
    axes[1, 0].set_ylim(0, 20)

    ### dtheta vs ellipticity
    ang_cut_min = cutoffs['min_dtheta']
    ang_cut_max = cutoffs['max_dtheta']

    ax = axes[0, 1]
    ax.axhline(ell_cut_max, c='k',
               lw=2, alpha=0.5)
    ax.axvline(ang_cut_min, c='k',
                    lw=2, alpha=0.5)
    ax.axvline(ang_cut_max, c='k',
                    lw=2, alpha=0.5)
    axes[0, 1].set_title('Size : Goodness of fit', fontsize=12)

    ## dtheta vs radius
    ax = axes[1, 1]
    ax.axhline(rad_cut_min, c='k',
                    lw=2, alpha=0.5)
    ax.axhline(rad_cut_max, c='k',
                    lw=2, alpha=0.5)
    ax.axvline(ang_cut_min, c='k',
                    lw=2, alpha=0.5)
    ax.axvline(ang_cut_max, c='k',
                    lw=2, alpha=0.5)
    axes[1, 1].set_xlabel('Angular aperture')
    #axes[1, 1].set_xlim(0, 1.)


def show_one_stack_output(stack, one_stack_output):
    '''
    Displays the result of `detection.one_stack_output`
    '''
    fig, axes = plt.subplots(stack.shape[0] // 3, 3,
                             sharex=True, sharey=True,
                             figsize=(9, 0.5*stack.shape[0]))
    for n, sub_labeled in enumerate(one_stack_output['labeled_stack']):
        try:
            ax = axes[n//3, n % 3]
        except IndexError:
            break
        ax.imshow(sub_labeled, cmap='gray')
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_xlim(0, sub_labeled.shape[0])
    ax.set_ylim(0, sub_labeled.shape[1])

    if 'all_props' in one_stack_output:
        for lbl, pos in one_stack_output['all_props'].iterrows():
            n = np.round(pos.z)
            try:
                ax = axes[n//3, n % 3]
            except IndexError:
                continue
            ax.plot(pos.y, pos.x, 'o',
                    mfc='b', alpha=0.5)

    if 'positions' in one_stack_output:
        for lbl, pos in one_stack_output['positions'].iterrows():
            n = np.round(pos.z)
            try:
                ax = axes[n//3, n % 3]
            except IndexError:
                continue
            ax.plot(pos.y, pos.x, 'o',
                    mec='r', mew=2, ms=10,
                    mfc='none', alpha=0.8)


def show_histogram(image, depth, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    bins = 2**depth
    h = ax.hist(image.flatten(),
                bins=bins-2, log=True)
    return h

def polar_histogram(cellcluster, ax=None, **kwargs):
    if ax is None:
        fig1, ax = plt.subplots(1, 1,
                                subplot_kw={'projection':'polar'},
                                figsize=(6,6))


    radii = cellcluster.theta_bin_count / np.float(cellcluster.theta_bin_count.sum())
    width = (cellcluster.theta_bins[1] - cellcluster.theta_bins[0])
    bars = ax.bar(cellcluster.theta_bins[:-1], radii,
                  width=width, bottom=0.0, **kwargs)
    #ax.set_rgrids([0.05, 0.1, 0.15,], angle=90)
    #ax.set_rmax(0.18)
    short_name = cellcluster.metadata['FileName'].split(os.path.sep)[-1].split('.')[0]
    ax.set_title(short_name)
    return ax

def show_overlayed(cellcluster, index, preprocess=None,
                   xy_ROI=None, ax=None, **kwargs):
    '''
    Show the stack number `index` with the detected positions overlayed
    '''

    z_stack = cellcluster.get_z_stack(index)
    if hasattr(cellcluster, 'preprocess'):
        z_stack = cellcluster.preprocess(z_stack)
    elif preprocess is not None:
        z_stack = preprocess(z_stack)
    ax = _show_overlayed(z_stack, cellcluster.trajs.loc[index],
                         cellcluster.metadata['PhysicalSizeX'],
                         cellcluster.metadata['PhysicalSizeZ'],
                         xy_ROI=xy_ROI, ax=ax, **kwargs)


def _show_overlayed(z_stack, positions,
                    xy_size, z_size,
                    xy_ROI=None,
                    ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    elif ax is not None:
        fig = ax.get_figure()

    if xy_ROI is not None:
        x_min, x_max, y_min, y_max = xy_ROI
        x_min = max(0, x_min)
        x_max = min(x_max, z_stack.shape[1])

        y_min = max(0, y_min)
        y_max = min(y_max, z_stack.shape[2])

        z_stack = z_stack[:, x_min:x_max, y_min:y_max]

    else:
        x_min, xmax, y_min, y_max = (0, z_stack.shape[-2],
                                     0, z_stack.shape[-1])

    # xy projection:
    if ax is None:
        ax_xy = fig.add_subplot(111)
    else:
        ax_xy = ax
    ax_xy.imshow(z_stack.max(axis=0))
    x_lim = ax_xy.get_xlim()
    y_lim = ax_xy.get_ylim()
    ax_xy.scatter(positions['y'] / xy_size - y_min,
                  positions['x'] / xy_size - x_min,
                  **kwargs)

    ax_xy.set_xlim(x_lim)
    ax_xy.set_ylim(y_lim)
    return ax_xy

def show_projected(cellcluster, index, preprocess=None,
                   xy_ROI=None, ax=None, **kwargs):
    '''
    Show the stack number `index` with the detected positions overlayed
    on maximum intensity projections of the stack in the 3 directions
    '''

    z_stack = cellcluster.get_z_stack(index)
    if hasattr(cellcluster, 'preprocess'):
        z_stack = cellcluster.preprocess(z_stack)
    elif preprocess is not None:
        z_stack = preprocess(z_stack)
    ax = _show_projected(z_stack, cellcluster.trajs.loc[index],
                         cellcluster.metadata['PhysicalSizeX'],
                         cellcluster.metadata['PhysicalSizeZ'],
                         xy_ROI=xy_ROI, ax=ax, **kwargs)


def _show_projected(z_stack, positions,
                    xy_size, z_size,
                    xy_ROI=None,
                    ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    elif ax is not None:
        fig = ax.get_figure()

    if xy_ROI is not None:
        x_min, x_max, y_min, y_max = xy_ROI
        x_min = max(0, x_min)
        x_max = min(x_max, z_stack.shape[1])

        y_min = max(0, y_min)
        y_max = min(y_max, z_stack.shape[2])

        z_stack = z_stack[:, x_min:x_max, y_min:y_max]

    else:
        x_min, xmax, y_min, y_max = (0, z_stack.shape[1],
                                     0, z_stack.shape[2])

    fig_width, fig_height = fig.get_size_inches()
    proj_height = fig_width * 0.3 # Allow for margins

    # xy projection:
    if ax is None:
        ax_xy = fig.add_subplot(111)
    else:
        ax_xy = ax
    ax_xy.imshow(z_stack.max(axis=0))
    x_lim = ax_xy.get_xlim()
    y_lim = ax_xy.get_ylim()
    ax_xy.scatter(positions['y'] / xy_size - y_min,
                  positions['x'] / xy_size - x_min,
                  **kwargs)
    divider = make_axes_locatable(ax_xy)
    ax_yz = divider.append_axes("top", proj_height,
                                pad=0.2, sharex=ax_xy)
    ax_yz.imshow(z_stack.max(axis=1),
                 aspect=z_size/xy_size)
    ax_yz.scatter(positions['y'] / xy_size  - y_min,
                  positions['z'] / z_size, **kwargs)

    ax_zx = divider.append_axes("right", proj_height,
                                pad=0.2, sharey=ax_xy)
    ax_zx.imshow(z_stack.max(axis=2).T,
                 aspect=xy_size/z_size)
    ax_zx.scatter(positions['z'] / z_size,
                  positions['x'] / xy_size - x_min,
                  **kwargs)
    z_lim = ax_yz.get_xlim()
    ax_yz.set_xlim(z_lim)
    ax_zx.set_ylim(z_lim)

    ax_xy.set_xlim(x_lim)
    ax_xy.set_ylim(y_lim)
    return ax_xy


def plot_rotation_events(cluster, ax=None,
                         show_segments=False,
                         **plot_kwargs):
    '''Displays the rotation event score

    Paramters
    ---------
    cluster : a Cellluster object
    ax : matplotlib Axes
    show_segments : bool
        if True, also show the rotation index for the individual segments
    plot_kwargs : dict
       keyword arguments passed to the plotting function
    '''

    if ax is None:
        fig, ax = plt.subplots()

    colors = cluster.trajs.get_colors()
    if show_segments:
        n_labels = cluster.trajs.labels.size

        for label in cluster.trajs.labels:
            try:
                sub_dr = cluster.detected_rotations.xs(label, level='label')
            except KeyError:
                continue
            color = colors[label]
            ts = sub_dr.index.values
            ts *= cluster.metadata['TimeIncrement']
            shift = (label - n_labels/2) / 100.
            ax.step(ts, sub_dr.values + shift, '-', c=color, lw=2, alpha=0.7)
    n_detected =  pd.Series(
        np.ones_like(cluster.detected_rotations),
        index=cluster.detected_rotations.index).sum(level='t_stamp').astype(np.float)
    total_rot = cluster.detected_rotations.sum(level='t_stamp') / n_detected
    ts = total_rot.index.get_level_values('t_stamp').values.copy()
    ts *= cluster.metadata['TimeIncrement']
    if plot_kwargs.get('c') is None:
        plot_kwargs['c'] = 'k'
    if plot_kwargs.get('lw') is None:
        plot_kwargs['lw'] = 2
    ax.step(ts, total_rot.values, '-', **plot_kwargs)

    ax.set_xlabel('Elapsed time (min)')
    ax.set_ylabel('Rotation events')
    ax.set_ylim(-0.1, 1.1)
    plt.draw()
    ax.set_title(cluster.metadata['FileName'])
    return ax, total_rot, n_detected


def show_4panel_ellipses(cluster, trajs,
                         label, sizes,  cutoffs,
                         method='polar',
                         scatter_kw={}, line_kw={},
                         ellipsis_kw={},
                         show_centers=False,
                         savefile=None, axes=None, ax_3d=None):

    coords=['x', 'y', 'z']

    axes, ax_3d = draw.show_4panels(trajs, label,
                                    axes=axes, ax_3d=ax_3d,
                                    scatter_kw=scatter_kw,
                                    line_kw=line_kw,
                                    coords=coords)
    for size in sizes:
        axes, ax_3d = show_ellipses(cluster, trajs,
                                    label, size,
                                    cutoffs=cutoffs,
                                    coords=coords,
                                    axes=axes, ax_3d=ax_3d,
                                    method=method,
                                    show_centers=show_centers,
                                    **ellipsis_kw)

    for ax in axes.flatten():
        ax.plot([0], [0], 'k+', ms=20)
    ax_3d.plot([0], [0], [0], 'k+', ms=20)
    if savefile is not None:
        plt.savefig(savefile)

    return axes, ax_3d


def show_ellipses(cluster, trajs,
                  label, size,
                  cutoffs=None,
                  coords=['x_r', 'y_r', 'z_r'],
                  axes=None, ax_3d=None,
                  method='polar',
                  show_centers=False,
                  return_lines=False,
                  **plot_kwargs):

    if axes is None:
        fig, axes = plt.subplots(2, 2,
                                 sharex='col',
                                 sharey='row')

        for ax in axes.ravel():
            ax.set_aspect('equal')
        axes[1, 1].axis('off')
        ax_3d = fig.add_subplot(224, projection='3d')

    segments = trajs.get_segments()
    all_data = cluster.ellipses.xs(size, level='size')
    data = all_data.xs(label, level='label')
    ellipses = Ellipses(size=size,
                        segment=segments[label],
                        data=data,
                        method=method,
                        coords=list(coords))
    if cutoffs is None:
        good_indexes = data[data['good'] == 1].index
    else:
        good_indexes = ellipses.good_indices(cutoffs)

    line_idx = []
    ell_idx = []
    lines_df = None
    for t_stamp in good_indexes:
        log.debug(
            'Good ellipse: size {}, label {}, time stamp {}'.format(size,
                                                                    label,
                                                                    t_stamp))
        curve = ellipses.evaluate(t_stamp)
        if curve is None:
            print('eval failed')
            raise
            #continue
        l_00 = axes[0, 0].plot(curve[:, 0], curve[:, 1], **plot_kwargs)[0]
        l_01 = axes[0, 1].plot(curve[:, 2], curve[:, 1], **plot_kwargs)[0]
        l_10 = axes[1, 0].plot(curve[:, 0], curve[:, 2], **plot_kwargs)[0]
        l_11 = ax_3d.plot(curve[:, 0], curve[:, 1], curve[:, 2], **plot_kwargs)[0]
        lines = (l_00, l_01, l_10, l_11)
        all_axes = (axes[0, 0], axes[0, 1], axes[1, 0], ax_3d)
        ### Store the correspondance between plotted lines and corresponding indexes
        for ax, line in zip(all_axes, lines):
            line_idx.append((ax, line))
            ell_idx.append([t_stamp, label, size])
    if len(line_idx):
        lines_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(line_idx,
                                                                names=['axis', 'line']),
                                data=np.array(ell_idx),
                                columns=['t_stamp', 'label', 'size'])

    if show_centers:
        axes[0, 0].plot(ellipses.data.loc[good_indexes]['x_ec'],
                        ellipses.data.loc[good_indexes]['y_ec'], 'r+')
        axes[0, 1].plot(ellipses.data.loc[good_indexes]['z_ec'],
                        ellipses.data.loc[good_indexes]['y_ec'], 'r+')
        axes[1, 0].plot(ellipses.data.loc[good_indexes]['x_ec'],
                        ellipses.data.loc[good_indexes]['z_ec'], 'r+')
    if return_lines:
        return axes, ax_3d, lines_df
    return axes, ax_3d

def show_n_panels(cluster, thumbs, time0,
                  n_panels=3, trail_length=10, fig=None):

    if fig is None:
        fig, axes = plt.subplots(1, n_panels,
                                 figsize = (4*n_panels, 4),
                                 sharex=True, sharey=True)
    else:
        axes = fig.get_axes()
    x_min, x_max, y_min, y_max = cluster.xy_ROI
    xy_size = cluster.metadata['PhysicalSizeX']
    data_dict = {}
    for n in range(n_panels):
        ax = axes[n]
        ax.clear()
        thumb = thumbs[time0+n, ...]
        ax.imshow(thumb)
        center = center_of_mass(thumb - thumb.min())
        ax.plot(center[1], center[0], 'o', ms=10, c='gray', alpha=0.5)
        upto_cur_pos = get_trail(cluster.trajs,
                                 trail_length, time0+n)
        if upto_cur_pos is None:
            continue
        labels = upto_cur_pos.index.get_level_values('label').unique()
        colors = cluster.trajs.get_colors()
        for label in labels:
            color = colors[label]
            upto = upto_cur_pos.xs(label, level='label')
            lines = ax.plot(upto.y / xy_size - y_min,
                            upto.x / xy_size - x_min, '-',
                            c=color, lw=4)
            if upto.index.get_level_values('t_stamp').values[-1] == time0 + n:
                ax.plot(upto.y.iloc[-1] / xy_size - y_min,
                        upto.x.iloc[-1] / xy_size - x_min, 's',
                        c=color, lw=4)
            data_dict[lines[0]] = label
        ax.set_title('Frame %i' % (time0 + n))
        ax.set_xlim(0, y_max-y_min)
        ax.set_ylim(x_max-x_min, 0)

    plt.draw()
    return fig, data_dict

def load_thumbs(cluster, reset_ROI=True,
                border=20, preprocess=None):

    xy_size = cluster.metadata['PhysicalSizeX']
    positions = cluster.trajs
    xy_ROI = (int(positions['x'].min() / xy_size) - border,
              int(positions['x'].max() / xy_size) + border,
              int(positions['y'].min() / xy_size) - border,
              int(positions['y'].max() / xy_size) + border)

    sizey = cluster.metadata['SizeY']
    sizex = cluster.metadata['SizeX']
    x_min, x_max, y_min, y_max = xy_ROI
    x_min = max(0, x_min)
    x_max = min(x_max, sizex)
    y_min = max(0, y_min)
    y_max = min(y_max, sizey)
    thumbs_shape = (cluster.metadata['SizeT'],
                    x_max - x_min, y_max - y_min)

    thumbs = np.zeros(thumbs_shape)
    if reset_ROI:
        cluster.xy_ROI = (x_min, x_max,
                          y_min, y_max)
    stack_iterator = build_iterator(cluster.stackio,
                                    preprocess)

    for t, stack in enumerate(stack_iterator()):
        if len(stack.shape) == 2:
            stack = stack[np.newaxis, ...]
        thumbs[t] = stack[:, x_min: x_max,
                          y_min: y_max].max(axis=0)
    return thumbs

def show_with_trail(time, upto_cur_pos,
                    z_stack, colors,
                    save_dir='.',
                    xy_size=1, z_size=1, aspect=1,
                    xy_ROI=None, save=True,
                    fig=None, ax=None):

    if len(z_stack.shape) == 2:
        z_stack = z_stack[np.newaxis, ...]

    if xy_ROI is not None:
        x_min, x_max, y_min, y_max = xy_ROI
        x_min = max(0, x_min)
        x_max = min(x_max, z_stack.shape[1])

        y_min = max(0, y_min)
        y_max = min(y_max, z_stack.shape[2])
    else:
        x_min, xmax, y_min, y_max = (0, z_stack.shape[1],
                                     0, z_stack.shape[2])
    if fig is None and ax is None:
        fig = plt.figure()
    elif ax is not None:
        fig = ax.get_figure()
    u_times = upto_cur_pos.index.get_level_values('t_stamp').unique()
    n_times = u_times.size
    if n_times == 1:
        cur_pos = upto_cur_pos
    else:
        cur_pos = upto_cur_pos.loc[u_times[-1]]
    labels = cur_pos.index.get_level_values('label')

    ax_xy = _show_overlayed(z_stack, cur_pos,
                            xy_size, z_size,
                            xy_ROI=xy_ROI,
                            colors=colors.values,
                            fig=fig, ax=ax)

    if upto_cur_pos.shape[0] > 1:
        for label in labels:
            color = colors.loc[label]
            upto = upto_cur_pos.xs(label, level='label')
            ax_xy.plot(upto.y / xy_size - y_min,
                       upto.x / xy_size - x_min,
                       c=color, ls='-', lw=4)
    if save:
        fig_name = os.path.join(save_dir,
                                'detected_%03i.png' % time)
        plt.draw()
        fig.savefig(fig_name)
        return 'Graph saved to {}'.format(fig_name)

def get_trail(trajs, trail, time):

    start = max(0, time-trail)
    times = trajs.index.get_level_values('t_stamp')
    start = times[times >= start][0]
    try:
        upto_cur_pos = trajs.loc[start: time]
    except KeyError:
        upto_cur_pos = None
    return upto_cur_pos

def upto_cur_pos_iter(trajs, trail):
    for time in trajs.index.get_level_values('t_stamp').unique():
        yield get_trail(trajs, trail, time)


######################################
# TODO port what's below
######################################

def make_movie(positions, iter_stacks, xy_size,
               z_size, save_dir, trail=0, aspect=1,
               engines=None, name='movie.avi', autozoom=True,
               save_thumbs=True):

    times = positions.index.get_level_values('t').unique()
    fig, ax = plt.subplots(figsize=(16, 12))
    if autozoom:
        xy_ROI = (int(positions['x'].min() / xy_size) - 5,
                  int(positions['x'].max() / xy_size) + 5,
                  int(positions['y'].min() / xy_size) - 5,
                  int(positions['y'].max() / xy_size) + 5)
    else:
        xy_ROI = None

    kwargs={'save_dir':save_dir,
            'xy_size':xy_size,
            'z_size': z_size,
            'aspect':aspect,
            'fig': fig,
            'xy_ROI': xy_ROI}
    arguments = zip(times,
                    upto_cur_pos_iter(positions, trail),
                    iter_stacks,
                    colors_iter(positions),
                    itertools.repeat(kwargs) )

    def mapper(args):
        (time, upto_cur_pos,
         z_stack, colors, kwargs) = args
        return show_with_trail(time, upto_cur_pos,
                               z_stack, colors, **kwargs)

    if engines is not None:
        results = engines.map(mapper, arguments)
    else:
        results = map(mapper, arguments)

    save_dir = kwargs['save_dir'].split(' ')
    if len(save_dir) == 1:
        save_dir = kwargs['save_dir']
    else:
        save_dir = '\ '.join(save_dir)

    done = 0
    for res in results:
        done += 1
    print('Registered %i files' %done)
    try:
        command = '''
        mencoder mf://{}/*.png\
        -mf w=800:h=600:fps=4:type=png\
        -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell\
        -oac copy -o {}'''.format(save_dir,
                                  os.path.join(save_dir, name))
        os.system(command)
    except OSError:
        warnings.warn("""Movie couldn't be compiled""")


class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def show_plane(ax, r0, r1, w, h, origin, **kwargs):
    '''
    draws a plane on a 3D axis

    Parameters:
    -----------
    r0, r1 : ndarrays, pair of (x, y, z) vectors
    w, h: ndarrays, pair on ints width and
        height of the plane in graph coordinates
    origin: ndarray (x, y, z) position of the center of the plane
    '''
    r0 /= (r0**2).sum()**0.5
    r1 /= (r1**2).sum()**0.5
    us, vs = np.mgrid[-w//2:w//2:10j, -h//2:h//2:10j]
    xs = us * r0[0] + vs * r1[0] + origin[0]
    ys = us * r0[1] + vs * r1[1] + origin[1]
    zs = us * r0[2] + vs * r1[2] + origin[2]
    ax.plot_surface(xs, ys, zs, **kwargs)

    plt.draw()
    return ax

def show_circle(ax, center, omega, point0):

    center = np.asarray(center)
    omega = np.asarray(omega)
    point0 = np.asarray(point0)

    thetas = np.linspace(0, 2*np.pi, 50)
    omega = omega / (omega**2).sum()**0.5
    r0 = point0 - center
    r1 = np.cross(omega, r0)

    circle_points = (np.vstack([r0[0] * np.cos(thetas),
                                r0[1] * np.cos(thetas),
                                r0[2] * np.cos(thetas)])
                     + np.vstack([r1[0] * np.sin(thetas),
                                  r1[1] * np.sin(thetas),
                                  r1[2] * np.sin(thetas)]))
    circle_points += center.repeat(50).reshape((3, 50))
    return circle_points

def colorcombine(red_im=None, green_im=None, blue_im=None, normalize=False):
    '''
    Creates an RGB image from grey level images.
    At least one of the three channels must be passed

    Parameters:
    -----------
    red_im: ndarray of shape (N, M) or None
        the image to be used as the red channel
    green_im: ndarray of shape (N, M) or None
        the image to be used as the green channel
    blue_im: ndarray of shape (N, M) or None
        the image to be used as the blue channel
    normalize: bool, optional, default False
        if True, the intensity in each channel will be scaled such that
        the lower intensity is 0 and the higher is 1 in the output

    Returns:
    --------
    rgb_image: nd array of shape (N, M, 3)
        the combine RGB image

    Example:
    --------
    >>> import numpy as np
    >>> from skimage.color import colorcombine
    >>> red = np.random.random((256, 256))
    >>> green = np.random.random((256, 256))
    >>> blue = np.random.random((256, 256))
    >>> rgb_im = colorcombine(red, green, blue)
    >>> rb_im = colorcombine(red, None, blue)
    '''
    channels = [red_im, green_im, blue_im]
    shape = None
    for channel in channels:
        if channel is not None:
            shape = channel.shape
            break
    if shape is None:
        raise ValueError('Provide at least one non-empty image')
    for n, channel in enumerate(channels):
        if channel is None:
            channel = np.zeros(shape)
        else:
            channel = img_as_float(channel)
            if normalize:
                channel -= channel.min()
                channel /= channel.max()
        channels[n] = channel
    return np.dstack(channels)

def center_traj_over_img(tracker, dsRed_dir, gfp_dir):

    rfp_list = load_img_list(os.path.join(tracker.metadata['data_path'],
                                          dsRed_dir))
    gfp_list = load_img_list(os.path.join(tracker.metadata['data_path'],
                                          gfp_dir))
    rfp0 = io.imread(rfp_list[0])
    gfp0 = io.imread(gfp_list[0])
    if len(rfp0.shape) == 3:
        rfp0 = rfp0.max(axis=0)
        gfp0 = gfp0.max(axis=0)
    elif len(rfp0.shape) == 4:
        rfp0 = rfp0.max(axis=0).max(axis=0)
        gfp0 = gfp0.max(axis=0).max(axis=0)


    rgb = colorcombine(rfp0, gfp0, None, normalize=True)
    fig, ax = plt.subplots()
    ax.imshow(rgb)
    ax.plot(tracker.center['y'] / tracker.metadata['xy_size'],
            tracker.center['x'] / tracker.metadata['xy_size'], 'w-', lw=2)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, rfp0.shape[1])
    ax.set_ylim(rfp0.shape[0], 0)
    return fig, ax

def show_measure(trajs, measure, errorbar=True,
                 show_segments=False, axes=None, **kwargs):
    data_cols = [col for col in measure if col != 't']
    n_measures = len(data_cols)
    if axes is None:
        fig, axes = plt.subplots(n_measures)
    else:
        fig = axes[0].get_figure()
    time = measure.t
    for col, sub_ax  in zip(data_cols, axes):
        data = measure[col]
        _show_single_measure(trajs, time, data,
                             errorbar=errorbar,
                             show_segments=show_segments,
                             ax=sub_ax, **kwargs)
        sub_ax.set_ylabel(col)
    sub_ax.set_xlabel('Time')

    return axes

def _show_single_measure(trajs, time, data,
                         errorbar=True, show_segments=True,
                         ax=None, **kwargs):

    if errorbar:
        ax.errorbar(time.mean(level='t_stamp'),
                    data.mean(level='t_stamp'),
                    yerr=data.std(level='t_stamp'), **kwargs)

    else:
        ax.plot(time.mean(level='t_stamp'),
                data.mean(level='t_stamp'),  **kwargs)
    if not show_segments:
        return ax
    colors = trajs.get_colors()
    for label in trajs.labels:
        sub_data = data.xs(label, level='label')
        sub_time = time.xs(label, level='label')
        ax.plot(sub_time, sub_data, '-', c=colors[label], alpha=0.2)
    return ax