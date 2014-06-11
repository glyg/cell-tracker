from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
import warnings

from sklearn.decomposition import PCA
from scipy.optimize import leastsq

from sktracker.trajectories import Trajectories
from sktracker.io import ObjectsIO, StackIO

from .conf import defaults
from .utils import continuous_theta

import logging
log = logging.getLogger(__name__)



ellipsis_cutoffs = defaults['ellipsis_cutoffs']

def scale(trajs, pix_sizex, pix_sizey,
          pix_sizez, inplace=False):

    out_trajs = trajs if inplace else trajs.copy()
    if pix_sizey == None:
        out_trajs[['x', 'y']] = trajs[['x', 'y']] * pix_sizex
    else:
        out_trajs['x'] = trajs['x'] * pix_sizex
        out_trajs['y'] = trajs['y'] * pix_sizey
    out_trajs['z'] = trajs['z'] * pix_sizez
    return out_trajs


class Ellipses():

    def __init__(self, size=0,
                 cutoffs=ellipsis_cutoffs,
                 segment=None,
                 data=None,
                 coords=['x_r', 'y_r', 'z_r']):
        self.coords = coords
        self.segment = segment
        self.size = size
        if data is not None:
            self.data = data
        elif segment is not None:
            self.do_fits(cutoffs)
        else:
            self.data = pd.DataFrame()

    def do_fits(self, cutoffs):

        param_names = ['a', 'b',# 'phi_x',
                       'phi_y','x0', 'y0']
        plane_components = ['r0_x', 'r0_y', 'r0_z',
                            'r1_x', 'r1_y', 'r1_z']
        columns = ['start', 'stop', 'ellipticity', 'chi2', 'dtheta',
                   'theta_i', 'theta_f', 'z']
        columns.extend(plane_components)
        columns.extend(param_names)

        self.data = pd.DataFrame(index=self.segment.index,
                                 columns=columns, dtype=np.float)
        if self.size > self.segment.t.ptp():
            warnings.warn('window size (%i) too big'
                          ' for segment with size %i\n'
                          % (self.size, self.segment.shape[0]))
            self.data['gof'] = - np.inf
            self.data['radius'] = -np.inf
            self.data['dtheta'] = 0.
            self.data['good'] =  np.zeros(self.data.shape[0])
            return self.data

        times  = self.segment.t
        indices = times[times < times.iloc[-1] - self.size].index

        #last = np.where(t_stamps > (t_stamps[-1] - self.size))[0][0]

        for idx in indices:
            start = idx[0]
            midle = times[times <= times.loc[idx] + self.size/2].index[-1][0]
            stop = times[times <= times.loc[idx] + self.size].index[-1][0]
            self.data.loc[midle, 'start'] = start
            self.data.loc[midle, 'stop'] = stop
            fit_output, components, rotated = fit_arc_ellipse(self.segment,
                                                              start, stop, self.coords,
                                                              return_rotated=True)
            if fit_output[-1] not in (1, 2, 3, 4):
                log.debug(
                    '''Fitting failed between {} and {} '''.format(start, stop))
                log.debug(fit_output[-2])
                continue
            self.data.loc[midle, 'z'] = rotated.z.loc[start:stop].mean()

            params = fit_output[0]
            #a, b, phi_x, phi_y, x0, y0 = params
            a, b, phi_y, x0, y0 = params
            a, b = np.abs([a, b])
            a, b = np.max((a, b)), np.min((a, b))
            self.data.loc[midle, 'ellipticity'] = a / b

            for n, p in zip(param_names, params):
                self.data.loc[midle][n] = p
            fvec = fit_output[2]['fvec']
            self.data.loc[midle, 'chi2'] = np.sum(fvec**2) / fvec.size
            r0 = components[0, :]
            r1 = components[1, :]
            for n, r in zip(plane_components[:3], r0):
                self.data.loc[midle, n] = r
            for n, r in zip(plane_components[3:], r1):
                self.data.loc[midle, n] = r
            thetas = np.arctan2(rotated.y.values - x0,
                                rotated.x.values - x0)
            dthetas, thetas = continuous_theta(thetas)
            self.data.loc[midle, 'theta_i'] = thetas.min()
            self.data.loc[midle, 'theta_f'] = thetas.max()
            self.data.loc[midle, 'dtheta'] = thetas.ptp()

        self.data['gof'] = - np.log(self.data['chi2'].astype(np.float))
        self.data['radius'] = self.data[['a', 'b']].abs().sum(axis=1) / 2.

        self.data['good'] =  np.zeros(self.data.shape[0])
        goods = self.good_indices(cutoffs)
        if len(goods):
            self.data['good'].loc[goods] = 1.
        # for col in columns:
        #     self.data[col] = self.data[col].astype(np.float)

    def evaluate(self, idx, sampling=4):


        sub_data = self.data.loc[idx]
        if not np.all(np.isfinite(sub_data)):
            return

        start, stop = sub_data[['start', 'stop']].astype(np.int)
        thetas = np.linspace(sub_data.theta_i,
                             sub_data.theta_f,
                             self.size*sampling)
        rhos = ellipse_radius(thetas, sub_data.a, sub_data.b,
                              sub_data.phi_y)
                              #sub_data.phi_x, sub_data.phi_y)

        xs = rhos * np.cos(thetas) + sub_data.x0
        ys = rhos * np.sin(thetas) + sub_data.y0
        zs = np.ones_like(rhos) * sub_data.z
        segdata = self.segment.loc[start:stop][self.coords]
        pca = PCA()
        pca.fit(segdata)
        ellipsis_fit = pca.inverse_transform(np.vstack((xs, ys, zs)).T)
        return ellipsis_fit

    def max_ellipticity(self, max_val):
        return self.data['ellipticity'].map(lambda x:
                                            x < max_val)

    def min_gof(self, min_val):
        return self.data['gof'].map(lambda x:
                                     x > min_val)

    def max_dtheta(self, max_val):
        max_val = max_val * np.pi / 180
        return self.data['dtheta'].map(lambda x:
                                       np.abs(x) < max_val)

    def min_dtheta(self, min_val):
        min_val = min_val * np.pi / 180
        return self.data['dtheta'].map(lambda x:
                                       np.abs(x) > min_val)

    def max_radius(self, max_val):
        return self.data['radius'].map(lambda x:
                                       x < max_val)

    def min_radius(self, min_val):
        return self.data['radius'].map(lambda x:
                                       x > min_val)


    def good_indices(self, cutoffs=ellipsis_cutoffs):

        max_e = cutoffs['max_ellipticity']
        min_g = cutoffs['min_gof']
        max_d = cutoffs['max_dtheta']
        min_d = cutoffs['min_dtheta']
        max_r = cutoffs['max_radius']
        min_r = cutoffs['min_radius']

        return self.data[self.max_ellipticity(max_e)
                         & self.min_gof(min_g)
                         & self.max_dtheta(max_d)
                         & self.min_dtheta(min_d)
                         & self.max_radius(max_r)
                         & self.min_radius(min_r)].index


def fit_arc_ellipse(segment, start, stop,
                    coords=['x_r', 'y_r', 'z_r'],
                    return_rotated=False):

    pca = PCA()
    sub_segment = segment[coords].loc[start:stop]
    if sub_segment.shape[0] < 4:
        warnings.warn('''Not enough points to fit an ellipse''')
        raise ValueError('''Not enough points to fit an ellipse''')
        #return None, None
    rotated = pca.fit_transform(sub_segment[coords])

    components = pca.components_
    to_fit = pd.DataFrame(data=rotated, index=sub_segment.index,
                          columns=('x', 'y', 'z'))
    # initial guesses
    a0 = to_fit['x'].ptp()
    b0 = to_fit['y'].ptp()
    phi_x0 , phi_y0 = 0., 0.
    x00, y00 = 0, 0
    #params0 = a0, b0, phi_x0, phi_y0, x00, y00
    params0 = a0, b0, phi_y0, x00, y00
    fit_output = leastsq(residuals, params0, [to_fit.x, to_fit.y],
                         full_output=1)
    if return_rotated:
        return fit_output, components, to_fit
    else:
        return fit_output, components


#def ellipse_radius(thetas, a, b, phi_x, phi_y):
def ellipse_radius(thetas, a, b, phi_y):
    ''' Computes the radius of an ellipse with
    the given paramters for the angles given by `thetas`

    Paramters
    ---------
    thetas: ndarray
        Angles in radians, in a polar coordiante system) for which the
        ellipses radius is to be computed
    a, b: floats
        The major and minor radii of the ellipsis
    phi_x, phi_y: floats
        Ellipsis phases along x and y, respectively
    '''
    rhos = a * b / np.sqrt((b * np.cos(thetas))**2
                           + (a * np.sin(thetas + phi_y))**2)

    return rhos

def residuals(params, data):
    '''

    '''
    #a, b, phi_x, phi_y, x0, y0 = params
    a, b, phi_y, x0, y0 = params
    x, y = data
    thetas = np.arctan2(y-y0, x-x0)
    rhos = np.hypot(x-x0, y-y0)
    #fit_rhos = ellipse_radius(thetas, a, b, phi_x, phi_y)
    fit_rhos = ellipse_radius(thetas, a, b, phi_y)
    return rhos - fit_rhos

