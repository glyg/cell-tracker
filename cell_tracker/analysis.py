import numpy as np
import pandas as pd
import warnings

from sklearn.decomposition import PCA
from scipy.optimize import leastsq

from sktracker.trajectories import Trajectories
from sktracker.io import ObjectsIO, StackIO

from . import ELLIPSIS_CUTOFFS

class Ellipses():

    def __init__(self, size=0,
                 cutoffs=ELLIPSIS_CUTOFFS,
                 segment=None,
                 data=None,
                 coords=['x_c', 'y_c', 'z_c']):
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

        param_names = ['a', 'b', 'omega', 'phi_x',
                       'phi_y','x0', 'y0']
        plane_components = ['r0_x', 'r0_y', 'r0_z',
                            'r1_x', 'r1_y', 'r1_z']
        columns = ['ellipticity', 'chi2']
        columns.extend(plane_components)
        columns.extend(param_names)

        self.data = pd.DataFrame(index=self.segment.index,
                                 columns=columns)
        if self.size > self.segment.shape[0]:
            warnings.warn('window size (%i) too big'
                          ' for segment with size %i\n'
                          % (self.size, self.segment.shape[0]))
            self.data['gof'] = - np.inf
            self.data['log_radius'] = -np.inf

            self.data['dtheta'] = 0.
            self.data['good'] =  np.zeros(self.data.shape[0])
            return self.data

        idxs = self.segment.index.get_level_values(level='t_stamp')
        last = np.where(idxs > idxs[-1] - self.size)[0][0]

        for i, idx in enumerate(self.segment.index[:last]):
            start = idx
            try:
                search_idx = idx[1]
            except:
                search_idx = idx

            stop = self.segment.index[idxs <= search_idx + self.size][-1]
            lstsq, components = fit_arc_ellipse(self.segment,
                                                start, stop,
                                                self.coords)
            if lstsq is None:
                continue
            params = lstsq[0]
            midle = self.segment.index[i + self.size//2]
            a, b = params[:2]
            a, b = np.abs(a), np.abs(b)
            a, b = np.max((a, b)), np.min((a, b))
            self.data.loc[midle]['ellipticity'] = a / b
            for n, p in zip(param_names, params):
                self.data.loc[midle][n] = p
                self.data.loc[midle]['chi2'] = (np.sum(lstsq[2]['fvec']**2)
                                                / self.size**2)
            r0 = components[0, :]
            r1 = components[1, :]
            for n, r in zip(plane_components[:3], r0):
                self.data.loc[midle][n] = r
            for n, r in zip(plane_components[3:], r1):
                self.data.loc[midle][n] = r

        self.data['gof'] = - np.log(self.data['chi2'].astype(np.float))
        self.data['log_radius'] = np.log(
            self.data[['a', 'b']].abs().sum(axis=1) / 2.)

        self.data['dtheta'] = self.data['omega'] * self.size

        self.data['good'] =  np.zeros(self.data.shape[0])
        goods = self.good_indices(cutoffs)
        if len(goods):
            self.data['good'].loc[goods] = 1.

    def evaluate(self, idx, sampling=10):

        start = idx - self.size//2
        if start < self.data.index[0]:
            return None
        stop = start + self.size
        params = self.data.loc[idx][['a', 'b', 'omega',
                                     'phi_x', 'phi_y',
                                     'x0', 'y0']]
        a, b, omega, phi_x, phi_y, x0, y0 = params
        r0 =  self.data.loc[idx][['r0_x', 'r0_y', 'r0_z']]
        r1 =  self.data.loc[idx][['r1_x', 'r1_y', 'r1_z']]
        r0 = np.asarray(r0)
        r1 = np.asarray(r1)
        n_pts = (stop - start) * sampling
        t = np.linspace(start, stop, n_pts)
        xs = a * np.cos(omega * t + phi_x) + x0
        ys = b * np.sin(omega * t + phi_y) + y0
        xs = np.vstack((xs,)*3)
        ys = np.vstack((ys,)*3)
        curve = r0[:, np.newaxis] * xs + r1[:, np.newaxis] * ys
        if self.segment is not None:
            s_center = self.segment[self.coords].loc[start:stop].mean(axis=0)
            c_center = curve.mean(axis=1)
            shift = s_center - c_center
            curve[0, :] += shift[0]
            curve[1, :] += shift[1]
            curve[2, :] += shift[2]
        return curve

    def max_ellipticity(self, max_val=3.):
        return self.data['ellipticity'].map(lambda x:
                                            x < max_val)

    def min_gof(self, min_val=1.):
        return self.data['gof'].map(lambda x:
                                     x > min_val)

    def max_dtheta(self, max_val=4*np.pi ):
        return self.data['dtheta'].map(lambda x:
                                       np.abs(x) < max_val)

    def min_dtheta(self, min_val=np.pi/6):
        return self.data['dtheta'].map(lambda x:
                                       np.abs(x) > min_val)

    def max_radius(self, max_val=60):
        return self.data['log_radius'].map(lambda x:
                                           np.exp(x) < max_val)

    def min_radius(self, min_val=5):
        return self.data['log_radius'].map(lambda x:
                                       np.exp(x) > min_val)


    def good_indices(self, cutoffs=ELLIPSIS_CUTOFFS):

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
                    coords=['x_c', 'y_c', 'z_c']):

    pca = PCA()
    sub_segment = segment[coords].loc[start:stop]
    if sub_segment.shape[0] < 4:
        warnings.warn('''Not enough points to fit an ellipse''')
        return None, None
    rotated = pca.fit_transform(sub_segment)
    center = np.array([0 for c in coords])
    r_center = pca.transform(center)[0]
    for n, coord in enumerate(coords):
        rotated[:, n] = rotated[:, n] - r_center[n]

    components = pca.components_
    to_fit = pd.DataFrame(data=rotated, index=sub_segment.index,
                          columns=('x', 'y', 'z'))
    # initial guesses
    a0 = to_fit['x'].ptp()
    b0 = to_fit['y'].ptp()
    omega0 = 1.#segment['ang_speed'+'_'.join(coords)].loc[start:stop].mean()
    phi_x0 = phi_y0 = 0.
    x00, y00 = to_fit.max(axis=0)[['x', 'y']]

    params0 = a0, b0, omega0, phi_x0, phi_y0, x00, y00
    lstsq = leastsq(arc_residuals, params0, args=(to_fit, ), full_output=1)
    return lstsq, components

def arc_ellipse(t, params):

    a, b, omega, phi_x, phi_y, x0, y0 = params
    #a, b, omega, phi_x, phi_y = params

    x = a * np.cos(omega*t + phi_x) + x0
    y = b * np.sin(omega*t + phi_y) + y0
    return x, y

def arc_residuals(params, data):

    times = np.asarray(data.index.get_level_values('t_stamp'), dtype=np.float)
    fit_x, fit_y = arc_ellipse(times, params)
    res_x = fit_x - data['x'].values
    res_y = fit_y - data['y'].values
    return np.hstack((res_y, res_x))
