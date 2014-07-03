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

fit_data_names = ['a', 'b', 'phi_y',
                  'x0', 'y0','z0',
                  'x_ec', 'y_ec', 'z_ec',
                  'ellipticity', 'radius',
                  'theta_i', 'theta_f', 'dtheta', 'omega',
                  'gof', 'fit_ier']

plane_components = ['r0_x', 'r0_y', 'r0_z',
                    'r1_x', 'r1_y', 'r1_z']

start_stop = ['start', 'stop', 'size']
columns = list(fit_data_names) # Make a copy (`copy` list attribute is python3 only)
columns.extend(plane_components)

columns.extend(start_stop)


def scale(trajs, pix_sizex, pix_sizey,
          pix_sizez, inplace=False):
    log.warning('''Deprecated, use `trajs.scale` instead''')
    return trajs.scale([pix_sizex, pix_sizey, pix_sizez], inplace=inplace)

class Ellipses():

    def __init__(self, size=0,
                 cutoffs=ellipsis_cutoffs,
                 method='polar',
                 segment=None,
                 data=None,
                 t_step=1.,
                 coords=['x_r', 'y_r', 'z_r']):
        self.coords = coords
        self.segment = segment
        self.size = size
        self.t_step = t_step
        self.method = method
        if data is not None:
            self.data = data
        elif segment is not None:
            self.do_fits(cutoffs)
        else:
            self.data = pd.DataFrame(index=self.segment.index,
                                     columns=columns, dtype=np.float)
        if not 'size' in self.data.columns:
            self.data['size'] = self.size

    def do_fits(self, cutoffs):

        self.data = pd.DataFrame(index=self.segment.index,
                                 columns=columns, dtype=np.float)
        if self.size > self.segment.t.ptp():
            log.debug('window size (%i) too big'
                      ' for segment with size %i\n'
                      % (self.size, self.segment.shape[0]))
            self.data['gof'] = - np.inf
            self.data['radius'] = -np.inf
            self.data['dtheta'] = 0.
            self.data['good'] =  np.zeros(self.data.shape[0])
            self.data['start'] = 0
            self.data['stop'] = 0
            return self.data

        times  = self.segment.t
        indices = times[times < times.iloc[-1] - self.size].index
        for idx in indices:
            start = idx[0]
            midle = times[times <= times.loc[idx] + self.size/2].index[-1][0]
            stop = times[times <= times.loc[idx] + self.size].index[-1][0]
            self.data.loc[midle, 'start'] = start
            self.data.loc[midle, 'stop'] = stop
            fit_data, components, rotated = fit_arc_ellipse(self.segment,
                                                              start, stop, self.coords,
                                                              method=self.method, ## 'polar' or 'cartesian'
                                                              return_rotated=True)
            if fit_data is None:
                log.debug(
                    '''Fitting failed between {} and {} '''.format(start, stop))
                continue

            for key, val in fit_data.iteritems():
                self.data.loc[midle, key] = val

            r0 = components[0, :]
            r1 = components[1, :]
            for n, r in zip(plane_components[:3], r0):
                self.data.loc[midle, n] = r
            for n, r in zip(plane_components[3:], r1):
                self.data.loc[midle, n] = r


        self.data['good'] =  np.zeros(self.data.shape[0])
        goods = self.good_indices(cutoffs)
        if len(goods):
            self.data['good'].loc[goods] = 1.

    def evaluate(self, idx, sampling=4):

        sub_data = self.data.loc[idx]
        if not np.all(np.isfinite(sub_data)):
            return

        start, stop = sub_data[['start', 'stop']].astype(np.int)
        if self.method == 'polar':
            thetas = np.linspace(sub_data.theta_i,
                                 sub_data.theta_f,
                                 self.size*sampling)
            rhos = ellipsis_radius(thetas, sub_data.a, sub_data.b,
                                   sub_data.phi_y)

            xs = rhos * np.cos(thetas) + sub_data.x0
            ys = rhos * np.sin(thetas) + sub_data.y0
        elif self.method == 'cartesian':
            t0 = np.int(self.segment.loc[start, 't'])
            t1 = np.int(self.segment.loc[stop, 't'])
            ts = np.linspace(t0, t1,
                             self.size*sampling)
            xs, ys = ellipsis_cartes(ts, sub_data.a, sub_data.b,
                                     sub_data.omega, sub_data.phi_x,
                                     sub_data.phi_y,
                                     sub_data.x0, sub_data.y0)

        zs = np.ones_like(xs) * sub_data.z0
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

        index = self.data[self.max_ellipticity(max_e)
                         & self.min_gof(min_g)
                         & self.max_dtheta(max_d)
                         & self.min_dtheta(min_d)
                         & self.max_radius(max_r)
                         & self.min_radius(min_r)].index
        log.debug(index)
        return index

def fit_arc_ellipse(segment, start, stop,
                    coords=['x_r', 'y_r', 'z_r'],
                    method='polar',
                    return_rotated=False):
    ''' Performs a least square fit of an arc ellipsis over the segment positions
    between 'start' and 'stop'

    Parameters
    ----------
    segment : pandas DataFrame
    start, stop : int, int
        bounds on which to perform the fit
    method : {'polar' | 'cartesian'}
        the fitting method (see Notes bellow)


    Notes
    -----
    This functin provides two fitting methods
    TODO : explain that

    '''


    pca = PCA()
    sub_segment = segment.loc[start:stop]
    if sub_segment.shape[0] < 6:
        log.debug('''Not enough points to fit an ellipsis''')
        if return_rotated:
            return None, None, None
        else:
            return None, None

    rotated = pca.fit_transform(sub_segment[coords])
    components = pca.components_
    to_fit = pd.DataFrame(data=rotated, index=sub_segment.index,
                          columns=('x', 'y', 'z'))
    to_fit['t'] = sub_segment.t

    # initial guesses
    a0 = to_fit['x'].ptp() / 2.
    b0 = to_fit['y'].ptp() / 2.
    phi_y0 = 0.
    x00, y00 = 0, 0
    params0 = [a0, b0, phi_y0, x00, y00]
    if method == 'polar':
        log.debug('Using polar method')
        fit_output = leastsq(residuals_polar, params0,
                             [to_fit.x, to_fit.y],
                             full_output=1)
    elif method == 'cartesian':
        log.debug('Using cartesian method')
        thetas = np.arctan2(to_fit.y, to_fit.x)
        dthetas, thetas = continuous_theta(thetas)
        omega0 = thetas.ptp() / to_fit.t.ptp()
        phi_x0 = 0
        params0.append(omega0)
        params0.append(phi_x0)
        fit_output = leastsq(residuals_cartesian, params0,
                             [to_fit.x, to_fit.y, to_fit.t],
                             full_output=1)
    else:
        raise ValueError('''method argument {} not understood
                         should be "polar" or "cartesian" '''.format(method))
    if fit_output[-1] not in (1, 2, 3, 4):
        log.debug(
            '''Leastsquare failed between {} and {}'''.format(start, stop))
        log.debug(fit_output[-2])
        if return_rotated:
            return None, None, None
        else:
            return None, None

    fit_data = pd.Series(index=[fit_data_names])
    params = fit_output[0]
    if method == 'polar':
        a, b, phi_y, x0, y0 = params
    elif method == 'cartesian':
        #a, b, phi_y, x0, y0, omega, phi_y_x = params
        a, b, phi_y, x0, y0, omega, phi_x = params
        x0 = x0# + r_center[0]
        y0 = y0# + r_center[1]
    ### Fit parameters
    fit_data['a'] = a
    fit_data['b'] = b
    fit_data['phi_y'] = phi_y
    if method == 'cartesian':
        fit_data['phi_x'] = phi_x
    else:
        fit_data['phi_x'] = phi_y
    fit_data['x0'] = x0
    fit_data['y0'] = y0
    z0 = to_fit.z.loc[start:stop].mean()
    fit_data['z0'] = z0

    ### Back projected ellipsis center geometrical data
    center_r = pca.inverse_transform([x0, y0, z0])
    fit_data['x_ec':'z_ec'] = center_r

    ### Ellipticity
    a, b = np.abs([a, b])
    fit_data['ellipticity'] = max(a, b) / min(a, b)
    fit_data['radius'] = (a + b) / 2


    ### Angular components
    thetas = np.arctan2(to_fit.y.values - y0,
                        to_fit.x.values - x0)
    dthetas, thetas = continuous_theta(thetas)
    fit_data['theta_i'] = thetas[0]
    fit_data['theta_f'] = thetas[-1]
    fit_data['dtheta'] = thetas[-1] - thetas[0]

    if method == 'polar':
        fit_data['omega'] = (fit_data['dtheta']
                             / (to_fit.t.loc[stop] - to_fit.t.loc[start]))
    elif method == 'cartesian':
        fit_data['omega'] = omega

    ### Goodness of fit
    fvec = fit_output[2]['fvec']
    fit_data['gof'] = -np.log(np.sum(fvec**2) / fvec.size)
    ### leastq info
    fit_data['fit_ier'] = fit_output[-1]

    if return_rotated:
        return fit_data, components, to_fit
    else:
        return fit_data, components


def residuals_cartesian(params, data):

    #a, b, phi_y, x0, y0, omega, phi_y_x = params
    a, b, phi_y, x0, y0, omega, phi_x = params
    x, y, t = data
    fit_x, fit_y = ellipsis_cartes(t, a, b, omega, phi_x, phi_y, x0, y0)
    res_x = x - fit_x
    res_y = y - fit_y
    return np.hstack((res_y, res_x))

def ellipsis_cartes(t, a, b, omega, phi_x, phi_y, x0, y0):

    thetas = omega * t
    x = a * np.cos(thetas + phi_x) + x0
    y = b * np.sin(thetas + phi_y) + y0
    return x, y

def residuals_polar(params, data):
    '''

    '''
    a, b, phi_y, x0, y0 = params
    x, y = data
    thetas = np.arctan2(y-y0, x-x0)
    rhos = np.hypot(x-x0, y-y0)
    fit_rhos = ellipsis_radius(thetas, a, b, phi_y)
    return rhos - fit_rhos

def ellipsis_radius(thetas, a, b, phi_y):
    ''' Computes the radius of an ellipsis with
    the given paramters for the angles given by `thetas`

    Paramters
    ---------
    thetas : ndarray
        Angles in radians, in a polar coordiante system) for which the
        ellipses radius is to be computed
    a, b : floats
        The major and minor radii of the ellipsis
    phi_y : float
        Ellipsis phase

    Returns
    -------

    rhos : ndarray
        The radii corresponding to the passed `thetas`
    '''
    rhos = a * b / np.sqrt((b * np.cos(thetas + phi_y))**2
                           + (a * np.sin(thetas + phi_y))**2)

    return rhos

