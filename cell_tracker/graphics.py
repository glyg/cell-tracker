# -*- coding: utf-8 -*-

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import warnings


#from .tracking import Ellipses

#draw a vector
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d, Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.ndimage import center_of_mass
from sktracker.trajectories import draw

import os
from .analysis import Ellipses

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
    ax.set_title(cellcluster.metadata['FileName'])
    return ax

def show_overlayed(cellcluster, index, xy_ROI=None, ax=None, **kwargs):
    '''
    Show the stack number `index` with the detected positions overlayed
    '''
    z_stack = cellcluster.stackio.get_tif_from_list(index).asarray()
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
        x_min, xmax, y_min, y_max = (0, z_stack.shape[1],
                                     0, z_stack.shape[2])

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

def show_projected(z_stack, positions,
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
        x_min, xmax, y_min, y_max = (0, z_stack.shape[1],
                                     0, z_stack.shape[2])

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
    ax_yz = divider.append_axes("top", 1.1,
                                pad=0.2, sharex=ax_xy)
    ax_yz.imshow(z_stack.max(axis=1),
                 aspect=z_size/xy_size)
    ax_yz.scatter(positions['y'] / xy_size  - y_min,
                  positions['z'] / z_size, **kwargs)

    ax_zx = divider.append_axes("right", 1.1,
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

    ax_xy = show_overlayed(z_stack, cur_pos,
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


def get_trail(positions, trail, time):

    start = max(0, time-trail)
    times = positions.index.get_level_values('t')
    start = times[times >= start][0]
    try:
        upto_cur_pos = positions.loc[start: time]
    except KeyError:
        upto_cur_pos = None
    return upto_cur_pos

def upto_cur_pos_iter(positions, trail):
    for time in positions.index.get_level_values('t').unique():
        yield get_trail(positions, trail, time)

def load_thumbs(tracker, xy_size, reset_ROI=True):

    border = np.int(tracker.metadata['min_radius']
                    / tracker.metadata['xy_size'])

    positions = tracker.track
    xy_ROI = (int(positions['x'].min() / xy_size) - border,
              int(positions['x'].max() / xy_size) + border,
              int(positions['y'].min() / xy_size) - border,
              int(positions['y'].max() / xy_size) + border)


    x_min, x_max, y_min, y_max = xy_ROI
    x_min = max(0, x_min)
    #x_max = min(x_max, z_stack.shape[1])
    y_min = max(0, y_min)
    #y_max = min(y_max, z_stack.shape[2])
    thumbs = []
    for stack in iter_stacks(tracker.metadata['data_path'],
                             preprocess):
        if len(stack.shape) == 2:
            stack = stack[np.newaxis, ...]

        if reset_ROI:
            x_max = min(x_max, stack.shape[1])
            y_max = min(y_max, stack.shape[2])
            tracker.xy_ROI = (x_min, x_max,
                              y_min, y_max)
            reset_ROI = False
        thumbs.append(stack[:, x_min: x_max,
                            y_min: y_max].max(axis=0))


    return np.array(thumbs)

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


def plot_rotation_events(cluster, ax=None,
                         show_segments=False, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    colors = cluster.trajs.get_colors()
    if show_segments:
        n_labels = cluster.trajs.labels.size

        for label in cluster.trajs.labels:
            sub_dr = cluster.detected_rotations.loc[label]
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
    ax.step(ts, total_rot.values, '-', c='k', lw=2)

    ax.set_xlabel('Elapsed time (min)')
    ax.set_ylabel('Rotation events')
    ax.set_ylim(-0.1, 1.1)
    plt.draw()
    ax.set_title(cluster.metadata['FileName'])
    return ax, total_rot, n_detected


def show_4panel_ellipses(cluster, label, sizes,  cutoffs,
                         savefile=None, axes=None, ax_3d=None):
    colors = cluster.trajs.get_colors()
    segment = cluster.trajs.get_segments()[label]
    scatter_kw = {'c':segment.t.astype(np.float),
                  'cmap':'spectral',
                  's':40,
                  'alpha':0.8,
                  'edgecolors':'none'}
    line_kw = {'c':'gray',#tracker.label_colors[label],
               'ls':'-',
               'alpha':0.8, 'lw':0.75}
    coords=['x_c', 'y_c', 'z_c']
    axes, ax_3d = draw.show_4panels(cluster.trajs, label,
                                    axes=axes, ax_3d=ax_3d,
                                    scatter_kw=scatter_kw,
                                    line_kw=line_kw,
                                    coords=coords)
    for size in sizes:
        axes, ax_3d = show_ellipses(cluster, label, size,
                                    cutoffs=cutoffs,
                                    coords=coords,
                                    axes=axes, ax_3d=ax_3d,
                                    alpha=0.5, lw=0.75, c='r')

    for ax in axes.flatten():
        ax.plot([0], [0], 'k+', ms=20)
    ax_3d.plot([0], [0], [0], 'k+', ms=20)
    if savefile is not None:
        plt.savefig(savefile)

    return axes, ax_3d


def show_ellipses(cluster, label, size,
                  cutoffs,
                  coords=['x_c', 'y_c', 'z_c'],
                  axes=None, ax_3d=None,
                  **plot_kwargs):

    if axes is None:
        fig, axes = plt.subplots(2, 2,
                                 sharex='col',
                                 sharey='row')

        for ax in axes.ravel():
            ax.set_aspect('equal')
        axes[1, 1].axis('off')
        ax_3d = fig.add_subplot(224, projection='3d')
    try:
        ellipses = Ellipses(size=size,
                            segment=cluster.interpolated.xs(label,
                                                            level='label'),
                            data=cluster.ellipses[size].xs(label,
                                                           level='label'),
                            coords=list(coords))
    except KeyError:
        return axes, ax_3d

    for idx in ellipses.good_indices(cutoffs):
        curve = ellipses.evaluate(idx)
        axes[0, 0].plot(curve[0, :], curve[1, :], **plot_kwargs)
        axes[0, 1].plot(curve[2, :], curve[1, :], **plot_kwargs)
        axes[1, 0].plot(curve[0, :], curve[2, :], **plot_kwargs)
        ax_3d.plot(curve[0, :], curve[1, :], curve[2, :], **plot_kwargs)
    return axes, ax_3d


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

def close_gap(tracker, label0, label1):

    old_labels = tracker.track.index.get_level_values('label').values
    old_labels[old_labels == label1] = label0
    tracker.track['new_label'] = old_labels
    tracker.track.set_index('new_label', append=True, inplace=True)
    tracker.track.reset_index(level='label', drop=True, inplace=True)
    try:
        tracker.track.index.set_names(['t', 'label'], inplace=True)
    except AttributeError:
        tracker.track.index.names = ['t', 'label']
    tracker.track.sortlevel('label', inplace=True)
    tracker.track.sortlevel('t', inplace=True)
    ## Remove duplicates... TODO: fix intensity and area
    grouped = tracker.track.groupby(level='t')
    tracker.track = grouped.apply(lambda df: df.mean(level='label'))

def cut_point(tracker, label, t):

    old_labels = tracker.track.index.get_level_values('label').values
    times = tracker.track.index.get_level_values('t').values
    points = (old_labels == label) * (times >= t)
    old_labels[points] = old_labels.max() + 1
    tracker.track['new_label'] = old_labels
    tracker.track.set_index('new_label', append=True, inplace=True)
    tracker.track.reset_index(level='label', drop=True, inplace=True)
    try:
        tracker.track.index.set_names(['t', 'label'], inplace=True)
    except AttributeError:
        tracker.track.index.names = ['t', 'label']
    tracker.track.sortlevel('label', inplace=True)
    tracker.track.sortlevel('t', inplace=True)



def show_n_pannels(tracker, thumbs, time0,
                   n_pannels=3, trail_length=10, fig=None):

    if fig is None:
        fig, axes = plt.subplots(1, n_pannels,
                                 figsize = (4*n_pannels, 4),
                                 sharex=True, sharey=True)
    else:
        axes = fig.get_axes()
    data_path = tracker.metadata['data_path']
    x_min, x_max, y_min, y_max = tracker.xy_ROI
    xy_size = tracker.metadata['xy_size']
    data_dict = {}
    for n in range(n_pannels):
        ax = axes[n]
        ax.clear()
        thumb = thumbs[time0+n, ...]
        ax.imshow(thumb)
        center = center_of_mass(thumb - thumb.min())
        ax.plot(center[1], center[0], 'o', ms=10, c='gray', alpha=0.5)
        upto_cur_pos = get_trail(tracker.track,
                                 trail_length, time0+n)
        if upto_cur_pos is None:
            continue
        labels = upto_cur_pos.index.get_level_values('label').unique()
        for label in labels:
            color = tracker.label_colors[label]
            upto = upto_cur_pos.xs(label, level='label')
            lines = ax.plot(upto.y / xy_size - y_min,
                            upto.x / xy_size - x_min, 'o-',
                            c=color, lw=4)
            data_dict[lines[0]] = label
        ax.set_title('Frame %i' % (time0 + n))
        ax.set_xlim(0, y_max-y_min)
        ax.set_ylim(x_max-x_min, 0)

    plt.draw()
    return fig, data_dict


class GapClose:
    '''
    '''
    def __init__(self, tracker,
                 t0=0, n_pannels=3, trail_length=10):

        self.t0 = t0
        self.n_pannels = n_pannels
        self.trail_length = trail_length
        self.tracker = tracker
        self.thumbs = load_thumbs(tracker,
                                  xy_size=tracker.metadata['xy_size'], reset_ROI=True)
        self.figure, self.data_dict = show_n_pannels(tracker, self.thumbs, t0,
                                                     self.n_pannels,
                                                     self.trail_length)
        self.canvas = self.figure.canvas
        self.cid = self.canvas.mpl_connect('button_press_event', self)
        self.cid2 = self.canvas.mpl_connect('key_press_event', self)
        self.xs = []
        self.ys = []
        self.to_close = []
        self.backups = []

    def __call__(self, event):

        if not hasattr(event, 'button'):
            if event.key == 'ctrl+z':
                if len(self.backups):
                    self.tracker.track = self.backups[-1]
                    self.__update__()

            elif event.key in [' ', 'right', 'up']:
                self.forward()
            elif event.key in ['left', 'down']:
                self.backward()
            elif event.key in ['c', 'C', 'enter']:
                if len(self.to_close) == 2:
                    self.backups.append(self.tracker.track.copy())
                    close_gap(self.tracker,
                              self.to_close[0],
                              self.to_close[1])
                    self.__update__()
                else:
                    event.inaxes.set_title('Please choose only two trajectories')
                    self.__update__()
            elif event.key in ['x', 'X']:
                if len(self.to_close) == 1:
                    self.backups.append(self.tracker.track.copy())
                    cut_point(self.tracker,
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

        self.figure, self.data_dict = show_n_pannels(
            self.tracker,
            self.thumbs, self.t0,
            self.n_pannels,
            trail_length=self.trail_length,
            fig=self.figure)
        self.to_close = []

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