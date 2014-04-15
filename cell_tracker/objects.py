import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import logging
log = logging.getLogger(__name__)

from sktracker.detection import nuclei_detector
from sktracker.trajectories import Trajectories
from sktracker.io import ObjectsIO, StackIO

from .detection import build_iterator
from .tracking import track_cells

from . import ELLIPSIS_CUTOFFS
from .analysis import Ellipses



class CellCluster:
    '''
    The container class for cell cluster migration analysis
    '''
    def __init__(self, stackio=None, objectsio=None):
        if stackio is not None:
            self.stackio = stackio
            if objectsio is None:
                self.oio = ObjectsIO.from_stackio(stackio)
        if objectsio is not None:
            self.oio = objectsio
            if stackio is None:
                self.stackio = StackIO.from_objectio(objectsio)
        try:
            self.trajs = Trajectories(self.oio['trajs'])
            log.info('Found trajectories in {}'.format(self.oio.store_path))
        except KeyError:
            pass

    @property
    def metadata(self):
        return self.oio.metadata

    def get_center(self, coords=['x', 'y', 'z'], smooth=0):
        """
        Computes `self.center`, the average positions (time stamp wise)
        adds columns with the passed coords appended with '_c' with the center
        positions
        """
        if not smooth:
            self.center = self.trajs[coords].mean(axis=0, level='t_stamp')
        else:
            interpolated = self.trajs.time_interpolate(coords=coords, s=smooth)
            self.center =  interpolated.mean(axis=0, level='stamp')
        new_cols = [c+'_c' for c in coords]
        self.trajs[new_cols] = self.reindexed_center()

    def detect_cells(self, preprocess, **kwargs):
        '''
        Detect cells and puts the positions in `self.trajs`
        '''
        if 'trajs' in self.oio.keys():
            self.oio['trajs.back'] = self.trajs
            log.warning('''Backed up former `trajs` data in
                        'trajs.back' ''')
        stack_iterator = build_iterator(self.stackio, preprocess)
        self.trajs = nuclei_detector(stack_iterator(),
                                     metadata=self.metadata,
                                     **kwargs)
        self.oio['trajs'] = self.trajs

    def track_cells(self, **kwargs):

        self.oio['trajs.back'] = self.trajs
        self.trajs = track_cells(self.trajs, **kwargs)
        self.oio['trajs'] = self.trajs

    def reindexed_center(self):
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

        rotated = self.pca.fit_transform(df[coords])
        for n, coord in enumerate(pca_coords):
            df[coord] = rotated[:, n]
        return df

    def cumulative_angle(self):
        '''
        Computes the angle of each cell with respect to the cluster center

        '''
        self.do_pca(coords=['x_c', 'y_c', 'z_c'])
        self.trajs['theta'] = np.arctan2(self.trajs['y_c_pca'],
                                         self.trajs['x_c_pca'])
        self.trajs['rho'] = np.hypot(self.trajs['y_c_pca'],
                                     self.trajs['x_c_pca'])

        self.trajs['dtheta'] = 0.
        grouped = self.trajs.groupby(level='label')
        tmp_trajs = grouped.apply(continuous_theta)
        tmp_trajs.index.set_names(['label', 't_stamp'],
                                   inplace=True)
        tmp_trajs = tmp_trajs.swaplevel('label', 't_stamp')
        tmp_trajs = tmp_trajs.sortlevel('t_stamp')
        self.trajs = Trajectories(tmp_trajs)
        self.theta_bin_count, self.theta_bins = np.histogram(
            self.trajs['theta'].dropna().astype(np.float),
            bins=np.linspace(-4*np.pi, 4*np.pi, 4 * 12 + 1))

        self.oio['trajs'] = self.trajs

    def compute_ellipticity(self, size=8,
                            cutoffs=ELLIPSIS_CUTOFFS,
                            coords=['x_c', 'y_c', 'z_c'], smooth=0):

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


def get_segment_rotations(segment, data):

    label = segment.index.get_level_values('label').unique()[0]
    t0 = segment.index.get_level_values('t_stamp')[0]
    t1 = segment.index.get_level_values('t_stamp')[-1]

    detected_rotations = pd.Series(data=0,
                                   index=pd.Index(np.arange(t0, t1), name='t_stamp'),
                                   name='detected_rotations')
    try:
        sub_data = data.xs(label, level='label')
    except IndexError:
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
        return None
    theta0 =  segment['theta'].iloc[0]
    segment['dtheta'] = segment['theta'].diff()
    segment['dtheta'].iloc[0] = 0
    segment['dtheta'][segment['dtheta'] > np.pi] -= 2 * np.pi
    segment['dtheta'][segment['dtheta'] < - np.pi] += 2 * np.pi
    segment['theta'] = segment['dtheta'].cumsum() + theta0
    segment.reset_index(level=1, drop=True, inplace=True)
    segment.reset_index(level=1, drop=True, inplace=True)
    segment.reindex_like(segment)
    segment['theta'].iloc[0] = theta0

    return segment
