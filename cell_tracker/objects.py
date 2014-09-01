# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import numpy as np
import pandas as pd

import logging
log = logging.getLogger(__name__)

from sktracker.detection import nuclei_detector
from sktracker.trajectories import Trajectories
from sktracker.trajectories.measures.transformation import do_pca
from sktracker.trajectories.measures.rotation import get_polar_coords

from sktracker.io import ObjectsIO, StackIO

from .tracking import track_cells

from .conf import defaults

ellipsis_cutoffs = defaults['ellipsis_cutoffs']
default_metadata = defaults['metadata']

from .analysis import Ellipses
from .utils import continuous_theta


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
            self.averages = pd.DataFrame(index=self.trajs.t_stamps)
            self.averages['t'] = self.trajs['t'].mean(level='t_stamp')

        except KeyError:
            pass
        try:
            self._complete_metadata()
        except KeyError:
            pass
        self._get_ellipses()
        ### Scaling lock
        self.was_scaled = True

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

    # def interpolate(self, sampling=1, s=0, k=3, backup=True):
    #     self.oio['raw'] = self.trajs
    #     self.trajs = self.trajs.time_interpolate(sampling=sampling, s=s, k=k)


    @property
    def metadata(self):
        return self.oio.metadata

    def _get_ellipses(self):

        self.ellipses = {}
        for key in self.oio.keys():
            if 'ellipses' in key:
                size = np.int(key.split('_')[-1])
                self.ellipses[size] = self.oio[key]

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

    def save_trajs(self, trajs_name='trajs'):
        self.oio['trajs_name'] = self.trajs
        if hasattr(self, 'averages'):
            self.oio['averages'] = self.averages


    def scale_pix_to_physical(self, coords=['x', 'y', 'z'], factors=None, force=False):
        ''' Scales the input data according to the metadata

        Parameters
        ----------

        coords : list of column indexes, optional, default ['x', 'y', 'z']
             the coordinates to scale
        factors : list of scaling factors
             if factors is None, defaults to self.metadata['PhysicalSize{X, Y, Z}']
        '''

        if factors is None:
            factors = [self.metadata['PhysicalSizeX'],
                       self.metadata['PhysicalSizeY'],
                       self.metadata['PhysicalSizeZ']][:len(coords)]

        if self.was_scaled and not force:
            raise ValueError('''It appears trajectories where allready scaled, use `force=True` '''
                             '''if you really want to do this''')

        self.trajs = self.trajs.scale(factors,
                                      coords=coords,
                                      inplace=True)
        self.was_scaled = True

    def scale_physical_to_pix(self, coords=['x', 'y', 'z'], shifts=None):
        '''Returns a copy of `self.trajs` scaled back to image space coordinates.

        Parameters
        ----------
        coords: column indexes

        '''
        trajs_pixels = self.trajs.scale([1/self.metadata['PhysicalSizeX'],
                                         1/self.metadata['PhysicalSizeY'],
                                         1/self.metadata['PhysicalSizeZ']],
                                        coords=coords,
                                        inplace=False)
        if shifts is not None:
            if len(shifts) != len(coords):
                raise ValueError('''Arguments shifts and coords should have the same length''')
            for shift, coord in zip(shifts, coords):
                trajs_pixels[coord] = trajs_pixels[coord] + shift
        return trajs_pixels


    def get_center(self, coords=['x', 'y', 'z'],
                   center_label=None,
                   smooth=0, reset=False,
                   append=True, trajs=None, save_droped='trajs'):
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

        if trajs is None:
            trajs = self.trajs

        if '/center' in self.oio.keys() and not reset:
            self.center = self.oio['center']
            log.info('Loading center trajcetory from HDFStore')
        elif center_label is not None:
            self.center = trajs[coords].xs(center_label, level='label')
            self.oio['center'] = self.center
            trajs = trajs.drop(center_label,
                               level='label',
                               inplace=False).sortlevel('t_stamp', inplace=False)
            trajs = Trajectories(trajs)
            log.info('Center trajectory dropped and wrote to HDFStore')
            self.oio['trajs_back'] = self.trajs
            self.oio[save_droped] = trajs
        elif not smooth:
            self.center = trajs[coords].mean(axis=0, level='t_stamp')
        else:
            interpolated = trajs.time_interpolate(coords=coords, s=smooth)
            self.center =  interpolated[coords].mean(axis=0, level='t_stamp')
        if append:
            new_cols = [c+'_c' for c in coords]
            trajs[new_cols] = self._reindexed_center(trajs)
            relative_coords = [c+'_r' for c in coords]
            trajs[relative_coords] = trajs[coords] - self._reindexed_center(trajs)
        if not hasattr(self, 'averages'):
            self.averages = pd.DataFrame(index=trajs.t_stamps)
            self.averages['t'] = trajs['t'].mean(level='t_stamp')
        for c in coords:
            self.averages[c] = self.center[c]
        return Trajectories(trajs.sortlevel(['t_stamp', 'label']))

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

    def _reindexed_center(self, trajs=None):
        if trajs is None:
            trajs = self.trajs
        return self.center.reindex(trajs.index, level='t_stamp')

    def do_pca(self, trajs=None, ndims=3,
               coords=['x', 'y', 'z'],
               suffix='_pca', append=True):
        '''
        Performs a principal component analysis on the input data
        '''
        if trajs is None:
            trajs = self.trajs
        rotated, self.pca = do_pca(trajs, pca=None,
                                   coords=coords,
                                   suffix=suffix,
                                   append=True, return_pca=True)
        return rotated

    def cumulative_angle(self, trajs=None):
        '''
        Computes the angle of each cell with respect to the cluster center

        '''
        if trajs is None:
            trajs = self.trajs
        #sself.trajs = Trajectories(self.trajs.dropna())
        #rotated = do_pca(self.trajs, coords=['x_r', 'y_r', 'z_r'])
        polar = get_polar_coords(trajs, get_dtheta=False, in_coords=['x_r', 'y_r'])

        trajs['theta'] = polar['theta']
        trajs['rho'] = polar['rho']
        self.theta_bin_count, self.theta_bins = np.histogram(
            trajs['theta'].dropna().astype(np.float),
            bins=np.linspace(-4*np.pi, 4*np.pi, 4 * 12 + 1))
        if not hasattr(self, 'averages'):
            self.averages = pd.DataFrame(index=trajs.t_stamps)
            self.averages['t'] = trajs['t'].mean(level='t_stamp')
        self.averages['theta'] = trajs['theta'].mean(level='t_stamp')
        return trajs
        #self.oio['trajs'] = self.trajs

    def compute_ellipticity(self, size=8, method='polar',
                            cutoffs=ellipsis_cutoffs, trajs=None,
                            sampling=1,
                            coords=['x_r', 'y_r', 'z_r']):

        if trajs is None:
            trajs = self.trajs
        grouped = trajs.groupby(level='label')
        t_step = self.metadata['TimeIncrement'] / sampling
        _ellipses = grouped.apply(evaluate_ellipticity,
                                  size=size, cutoffs=cutoffs,
                                  method=method,
                                  coords=coords,
                                  t_step=t_step)
        self.ellipses[size] = _ellipses.sortlevel(level='t_stamp')
        self.oio['ellipses_%i' %size] = self.ellipses[size]

    def detect_rotations(self, cutoffs, sizes, method='binary', trajs=None):
        if trajs is None:
            trajs = self.trajs

        for size in sizes:
            ellipsis_df = self.ellipses[size]
            ellipsis = Ellipses(size, data=ellipsis_df)
            ellipsis.data['good'] =  np.zeros(ellipsis.data.shape[0])
            goods = ellipsis.good_indices(cutoffs)
            if len(goods):
                ellipsis_df.loc[goods, 'good'] = 1.
            ellipsis_df['size'] = size

        data = pd.concat([self.ellipses[size][['start', 'stop', 'gof', 'radius',
                                               'dtheta', 'good', 'size']].astype(np.float)
                          for size in sizes]).dropna()
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        data.sort_index(axis=0, inplace=True)
        data.sort_index(axis=1, inplace=True)
        detected_rotations = trajs.groupby(
            level='label', group_keys=False).apply(_get_segment_rotations, data, method)
        if detected_rotations.ndim == 2:
            detected_rotations = detected_rotations.stack()
        detected_rotations = detected_rotations.swaplevel('label', 't_stamp')
        detected_rotations = detected_rotations.sortlevel(level=['t_stamp', 'label'])
        self.detected_rotations = detected_rotations


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

def _get_segment_rotations(segment, data, method):
    '''
    Paramters
    ---------
    method : {'binary' | 'score'}
       the method used to compute the score
    '''
    label = segment.index.get_level_values('label').unique()[0]
    t0 = segment.index.get_level_values('t_stamp')[0]
    t1 = segment.index.get_level_values('t_stamp')[-1]

    detected_rotations = pd.Series(data=0,
                                   index=segment.index, #pd.Index(np.arange(t0, t1), name='t_stamp'),
                                   name='detected_rotations')
    try:
        sub_data = data.xs(label, level='label')
    except KeyError:
        detected_rotations.loc[:] = np.nan
        return detected_rotations

    sizes = data['size'].unique()
    n_sizes = len(sizes)
    for size in sizes:
        at_size = sub_data[sub_data['size'] == size]
        good = at_size[at_size['good'] == 1]
        if good.shape[0] == 0:
            continue
        else:
            for t_stamp in good.index:
                start = sub_data[sub_data['size'] == size].loc[t_stamp, 'start']
                stop =  sub_data[sub_data['size'] == size].loc[t_stamp, 'stop']
                if method == 'score':
                    detected_rotations.loc[np.int(start): np.int(stop)] += 1. / n_sizes
                elif method == 'binary':
                    detected_rotations.loc[np.int(start): np.int(stop)] = 1
                else:
                    raise ValueError('''method should be either "binary" or "score"''')
    return detected_rotations


def evaluate_ellipticity(segment, size=0,
                         cutoffs=ellipsis_cutoffs,
                         method='polar',
                         data=None,
                         coords=['x_r', 'y_r', 'z_r'],
                         t_step=1):
    '''
    Fits an ellipse over a windows of size `size` in minutes
    '''
    ellipses = Ellipses(segment=segment, size=size,
                        cutoffs=ellipsis_cutoffs,
                        method=method,
                        data=None,
                        t_step=t_step,
                        coords=coords)
    return ellipses.data

def continuous_theta_(segment):
    '''
    Computes a continuous angle from a :math:`2*\pi` periodic one
    '''
    dthetas, thetas = continuous_theta(segment.theta.values)
    segment['dtheta'] = dthetas
    segment['theta'] = thetas
    return segment

#def forward_vec()
