# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import logging
log = logging.getLogger(__name__)

from sktracker.detection import nuclei_detector
from sktracker.trajectories import Trajectories
from sktracker.io import ObjectsIO, StackIO

from .tracking import track_cells

from .conf import defaults

ellipsis_cutoffs = defaults['ellipsis_cutoffs']
default_metadata = defaults['metadata']

from .analysis import Ellipses



class CellCluster:
    '''
    The container class for cell cluster migration analysis
    '''
    def __init__(self, stackio=None, objectsio=None):
        '''

        Parameters
        ----------

        stackio : a :class:`sktracker.StackIO` instance
        objectsio : a :class:`sktracker.ObjectsIO` instance

        '''
        if stackio is not None:
            self.stackio = stackio
            if objectsio is None:
                self.oio = ObjectsIO.from_stackio(stackio)
        if objectsio is not None:
            self.oio = objectsio
            if stackio is None:
                self.stackio = StackIO.from_objectsio(objectsio)
        try:
            self.trajs = Trajectories(self.oio['trajs'])
            log.info('Found trajectories in {}'.format(self.oio.store_path))
        except KeyError:
            pass
        self._complete_metadata()

    def _complete_metadata(self):
        shape = self.metadata['Shape']

        dim_order = self.metadata['DimensionOrder']
        dim_order = dim_order.replace('I', 'Z')
        self.metadata['DimensionOrder'] = dim_order
        for dim_label in dim_order:
            try:
                dim_id = dim_order.index(dim_label)
                self.metadata["Size" + dim_label] = shape[dim_id]
            except:
                self.metadata["Size" + dim_label] = 1
        for key, val in default_metadata.items():
            if key not in self.metadata:
                self.metadata[key] = val

    @property
    def metadata(self):
        return self.oio.metadata

    def get_z_stack(self, stack_num):
        """
        Returns the z stack at the time stamp
        given by stack_num

        For convenience, if this is a 2D image, it translates
        it to 3D
        """
        if self.stackio.image_path_list is not None:
            z_stack = self.stackio.get_tif_from_list(stack_num).asarray()
        else:
            z_stack = self.stackio.get_tif().asarray()[stack_num, ...]
        if len(z_stack.shape) == 2:
            z_stack = z_stack[np.newaxis, ...]
        return z_stack

    def get_center(self, coords=['x', 'y', 'z'], smooth=0,
                   append=True, relative=True):
        """Computes `self.center`, the average positions (time stamp wise).

        If `append` is True, appends columns named after the passed
        coordinates suffixed with '_c', containing the center
        positions.

        Parameters
        ----------

        coords : list of str, default `['x', 'y', 'z']`
            The coordinates on which to compute the center positions

        smooth : float, default 0.
            A smoothing factor. If non zero, the argument is passed
            to the `time_interpolate` method of `self.trajs`

        append: bool, default True
            If True, creates columns in `self.trajs` with the coordinates names
            suffixed with '_c' with the repeated  center postion and columns
            suffixed with `_r` with the positions relative to the center

        See Also
        --------

        sktracker.Trajectories.time_interpolate
        """
        if not smooth:
            self.center = self.trajs[coords].mean(axis=0, level='t_stamp')
        else:
            interpolated = self.trajs.time_interpolate(coords=coords, s=smooth)
            self.center =  interpolated[coords].mean(axis=0, level='t_stamp')
        if append:
            new_cols = [c+'_c' for c in coords]
            self.trajs[new_cols] = self._reindexed_center()
        if relative:
            relative_coords = [c+'_r' for c in coords]
            self.trajs[relative_coords] = self.trajs[coords] - self._reindexed_center()

    def detect_cells(self, preprocess, **kwargs):
        '''
        Detect cells and puts the positions in `self.trajs`
        '''
        if 'trajs' in self.oio.keys():
            self.oio['trajs.back'] = self.trajs
            log.warning('''Backed up former `trajs` data in
                        'trajs.back' ''')
        self.stack_iterator = build_iterator(self.stackio, preprocess)
        self.trajs = nuclei_detector(self.stack_iterator(),
                                     metadata=self.metadata,
                                     **kwargs)
        self.oio['trajs'] = self.trajs

    def track_cells(self, **kwargs):

        self.oio['trajs.back'] = self.trajs
        self.trajs = track_cells(self.trajs, **kwargs)
        self.oio['trajs'] = self.trajs

    def _reindexed_center(self):
        return self.center.reindex(self.trajs.index, level='t_stamp')

    def do_pca(self, df=None, ndims=3,
               coords=['x', 'y', 'z'],
               suffix='_pca'):
        '''
        Performs a principal component analysis on the input data
        '''
        if not df:
            df = self.trajs
        self.pca = PCA()
        pca_coords = [c + suffix for c in coords]
        if ndims == 2:
            coords = coords[:2]
            pca_coords = pca_coords[:2]
        try:
            rotated = self.pca.fit_transform(df[coords])
        except ValueError:
            raise ('''Remove non finite values before you attempt to perform PCA''')
        for n, coord in enumerate(pca_coords):
            df[coord] = rotated[:, n]
        return df

    def cumulative_angle(self):
        '''
        Computes the angle of each cell with respect to the cluster center

        '''
        self.do_pca(coords=['x_r', 'y_r', 'z_r'])
        self.trajs['theta'] = np.arctan2(self.trajs['y_r_pca'],
                                         self.trajs['x_r_pca'])
        self.trajs['rho'] = np.hypot(self.trajs['y_r_pca'],
                                     self.trajs['x_r_pca'])

        grouped = self.trajs.groupby(level='label', as_index=False)
        tmp_trajs = grouped.apply(continuous_theta)
        tmp_trajs = tmp_trajs.sortlevel('t_stamp')
        self.trajs = Trajectories(tmp_trajs)
        self.theta_bin_count, self.theta_bins = np.histogram(
            self.trajs['theta'].dropna().astype(np.float),
            bins=np.linspace(-4*np.pi, 4*np.pi, 4 * 12 + 1))

        #self.oio['trajs'] = self.trajs

    def compute_ellipticity(self, size=8,
                            cutoffs=ellipsis_cutoffs,
                            coords=['x_r', 'y_r', 'z_r'], smooth=0):
        size = np.int(size / self.metadata['TimeIncrement'])
        if not hasattr(self, 'ellipses'):
            self.ellipses = {}
        if not hasattr(self, 'interpolated'):
            self.interpolated = self.trajs.time_interpolate(coords=coords,
                                                            s=smooth)

        grouped = self.interpolated.groupby(level='label')
        self.ellipses[size] = grouped.apply(evaluate_ellipticity,
                                            size=size, cutoffs=cutoffs,
                                            coords=coords)
        self.oio['ellipses_%i' %size] = self.ellipses[size]

    def detect_rotations(self, cutoffs):

        for size in self.ellipses.keys():
            ellipsis_df = self.ellipses[size]
            ellipsis = Ellipses(size, data=ellipsis_df)
            ellipsis.data['good'] =  np.zeros(ellipsis.data.shape[0])
            goods = ellipsis.good_indices(cutoffs)
            if len(goods):
                ellipsis.data['good'].loc[goods] = 1.
            ellipsis.data['size'] = size

        data = pd.concat([ellipsis[['gof', 'log_radius',
                                    'dtheta', 'good', 'size']].astype(np.float)
                      for ellipsis in self.ellipses.values()]).dropna()
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        data = data.swaplevel('t_stamp', 'label')
        data.sort_index(axis=0, inplace=True)
        data.sort_index(axis=1, inplace=True)
        self.detected_rotations = self.trajs.groupby(
            level='label').apply(get_segment_rotations, data)

def build_iterator(stackio, preprocess=None):

    is_list = False
    if stackio.image_path_list is not None:
        if len(stackio.image_path_list) > 1:
            is_list = True
    if is_list:
        base_iterator = stackio.list_iterator()
    else:
        if stackio.metadata['DimensionOrder'] in ('TZCXY', 'TICXY',
                                                  'TZCYX', 'TICYX'):
            base_iterator = stackio.image_iterator(-4)
        elif stackio.metadata['DimensionOrder'] in ('TZXY', 'TZYX', 'TIXY', 'TIYX'):
            base_iterator = stackio.image_iterator(-3)
        elif stackio.metadata['DimensionOrder'] in ('TXY', 'TYX', 'IXY', 'IYX'):
            base_iterator = stackio.image_iterator(-2)
        else:
            raise TypeError('Stack organisation not understood')
    if preprocess is None:
        iterator = base_iterator
    else:
        def iterator():
            for stack in base_iterator():
                yield preprocess(stack)
    return iterator

def get_segment_rotations(segment, data):

    label = segment.index.get_level_values('label').unique()[0]
    t0 = segment.index.get_level_values('t_stamp')[0]
    t1 = segment.index.get_level_values('t_stamp')[-1]

    detected_rotations = pd.Series(data=0,
                                   index=pd.Index(np.arange(t0, t1), name='t_stamp'),
                                   name='detected_rotations')
    try:
        sub_data = data.loc[label]
    except KeyError:
        detected_rotations[:] = np.nan
        return detected_rotations
    sub_time = sub_data.index.values
    sizes = data['size'].unique()

    for size in sizes:
        at_size = sub_data[sub_data['size'] == size]
        good = at_size[at_size['good'] == 1]
        if good.shape[0] == 0:
            continue
        else:
            for t in good.index:
                start, stop = np.int(t-size//2), np.int(t+size//2)
                detected_rotations.loc[start: stop] = 1
    return detected_rotations


def evaluate_ellipticity(segment, **kwargs):
    '''
    Fits an ellipse over a windows of size `size`
    '''
    ellipses = Ellipses(segment=segment, **kwargs).data
    return ellipses


def continuous_theta(segment):
    '''
    Computes a continuous angle from a 2*np.pi periodic one
    '''
    if segment.shape[0] == 1:
        segment['dtheta'] = 0
        return segment
    theta0 =  segment['theta'].iloc[0]
    segment['dtheta'] = segment['theta'].diff()
    segment['dtheta'].iloc[0] = 0
    segment['dtheta'][segment['dtheta'] > np.pi] -= 2 * np.pi
    segment['dtheta'][segment['dtheta'] < - np.pi] += 2 * np.pi
    segment['theta'] = segment['dtheta'].cumsum() + theta0
    return segment
