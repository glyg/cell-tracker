# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import numpy as np
import pandas as pd
from sktracker.trajectories import Trajectories

def shifted_dif(df, coords, shift):
    left_shift = -np.floor(shift/2).astype(np.int)
    right_shift = np.ceil(shift/2).astype(np.int)
    return df[coords].shift(left_shift) - df[coords].shift(right_shift)


def p2p_dif(segment, coords, t_stamp0, t_stamp1):
    return segment[coords].loc[t_stamp1] - segment[coords].loc[t_stamp0]


def p2p_cum_directionality(cluster, t_stamp0, t_stamp1):
    #window = int(window / cluster.metadata['TimeIncrement'])
    shifted = cluster.trajs.groupby(level='label',
                                    group_keys=False).apply(p2p_dif,
                                                            ['x_pca', 'disp'],
                                                            t_stamp0, t_stamp1)
    shifted['data'] = shifted['x_pca'] / shifted['disp']
    return shifted['data']



def p2p_directionality(cluster, t_stamp0, t_stamp1):
    shifted = cluster.trajs.groupby(level='label',
                                    group_keys=False).apply(p2p_dif,
                                                            ['x_pca', 'y_pca', 'z_pca'],
                                                            t_stamp0, t_stamp1)

    shifted['data'] = (shifted.x_pca /
                       np.linalg.norm(shifted[['x_pca', 'y_pca', 'z_pca']],
                                      axis=1))
    return shifted['data']

def p2p_processivity(cluster, t_stamp0, t_stamp1, signed=True):

    cluster.trajs = Trajectories(
        cluster.trajs.groupby(level='label').apply(compute_displacement,
                                                   coords=['x', 'y', 'z']))
    shifted = cluster.trajs.groupby(level='label',
                                    group_keys=False).apply(p2p_dif,
                                                            ['x_pca', 'y_pca', 'z_pca', 'disp'],
                                                            t_stamp0, t_stamp1)
    if signed:
        shifted['data'] = (np.linalg.norm(shifted[['x_pca', 'y_pca', 'z_pca']], axis=1) /
                                   shifted['disp']) * np.sign(shifted['x_pca'])
    else:
        shifted['data'] = (np.linalg.norm(shifted[['x_pca', 'y_pca', 'z_pca']], axis=1) /
                                   shifted['disp'])
    return shifted['data']

def directionality(cluster, window):
    shifted = cluster.trajs.groupby(level='label').apply(shifted_dif,
                                                         ['x_pca', 'y_pca', 'z_pca'],
                                                         window)

    shifted['t'] = cluster.trajs.t
    shifted['data'] = (shifted.x_pca /
                         np.linalg.norm(shifted[['x_pca', 'y_pca', 'z_pca']],
                                        axis=1))
    return shifted[['t', 'data']]

def processivity(cluster, window, signed=True):
    cluster.trajs = Trajectories(
        cluster.trajs.groupby(level='label').apply(compute_displacement,
                                                   coords=['x', 'y', 'z']))
    shifted = cluster.trajs.groupby(level='label').apply(shifted_dif,
                                                         ['x_pca', 'y_pca', 'z_pca', 'disp'], window)
    if signed:
        shifted['data'] = (np.linalg.norm(shifted[['x_pca', 'y_pca', 'z_pca']], axis=1) /
                                   shifted['disp']) * np.sign(shifted['x_pca'])
    else:
        shifted['data'] = (np.linalg.norm(shifted[['x_pca', 'y_pca', 'z_pca']], axis=1) /
                                   shifted['disp'])
    shifted['t'] = cluster.trajs.t
    return shifted[['t', 'data']]

def cumulative_directionality(cluster, window):
    #window = int(window / cluster.metadata['TimeIncrement'])
    cluster.trajs = Trajectories(
        cluster.trajs.groupby(level='label').apply(compute_displacement,
                                                   coords=['x', 'y', 'z']))
    if not hasattr(cluster.trajs, 'x_pca'):
        cluster.do_pca()
    shifted = cluster.trajs.groupby(level='label').apply(shifted_dif,
                                                         ['x_pca', 'disp'], window)
    shifted['data'] = shifted['x_pca'] / shifted['disp']
    shifted['t'] = cluster.trajs.t
    return shifted[['t', 'data']]

def forward_displacement(cluster):

    cluster.do_pca()
    print('''Fraction of the movement's variance along the principal axis : {0:.1f}%'''
          .format(cluster.pca.explained_variance_ratio_[0] * 100))
    cluster.trajs = Trajectories(
        cluster.trajs.groupby(level='label').apply(compute_displacement,
                                                coords=['x_pca', 'y_pca', 'z_pca']))

def get_MSD(cluster):
    '''
    Compute the mean square displacement for each segment
    '''
    dts = cluster.trajs.t_stamps - cluster.trajs.t_stamps[0]
    MSD = cluster.trajs.groupby(level='label').apply(compute_MSD, dts)
    MSD['Dt'] = MSD.index.get_level_values('Dt_stamp') * cluster.metadata['TimeIncrement']
    return MSD


def compute_displacement(segment, coords=['x', 'y', 'z']):
    '''Computes the cumulated displacement of the segment given by

    .. math::
    \begin{aligned}
    D(0) &= 0\\
    D(t) &= \sum_{i=1}^{t} \left((x_i - x_{i-1})^2 + (y_i - y_{i-1})^2 + (z_i - z_{i-1})^2\right)^{1/2}\\
    \end{aligned}

    '''
    x, y, z = coords
    displacement = np.sqrt(segment[x].diff()**2
                           + segment[y].diff()**2
                           + segment[z].diff()**2)
    displacement = displacement.cumsum()
    displacement.iloc[0] = 0
    segment['disp'] = displacement
    #segment['fwd_frac'] = (segment[x] - segment[x].iloc[0]) / segment['disp']
    return segment

def compute_MSD(segment, dts, coords=['x', 'y', 'z']):
    '''Computes the mean square displacement of the segment given by

    .. math::
    \begin{aligned}
    \mbox{MSD}(\Delta t) &=  \frac{\sum_0^{T - \Delta t}
        \left(\mathbf{r}(t + \Delta t)  - \mathbf{r}(t) \right)^2}{(T - \Delta t) / \delta t}
    \end{aligned}
    '''
    dts = np.asarray(dts, dtype=np.int)
    msds = pd.DataFrame(index=pd.Index(dts, name='Dt_stamp'),
                        columns=['MSD', 'MSD_std'], dtype=np.float)
    msds.loc[0] = 0, 0
    for dt in dts[1:]:
        msd = ((segment[coords]
                - segment[coords].shift(dt)).dropna()**2).sum(axis=1)
        msds.loc[dt, 'MSD'] = msd.mean()
        msds.loc[dt, 'MSD_std'] = msd.std()
    return msds

