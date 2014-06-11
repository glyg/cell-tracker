import numpy as np
import pandas as pd
from cell_tracker.analysis import fit_arc_ellipse


def get_mock_segment(radius=30, dtheta=4*np.pi/3,
                     ellipticity=1.5,
                     noise=1e-4, n_points=10,
                     t_span=90, t_step=3, rotation=None):

    a = radius * ellipticity
    b = radius / ellipticity

    phi_y, x0, y0 = 0, 0, 0

    thetas = np.linspace(0, dtheta, n_points)
    xs = a * np.cos(thetas)
    ys = b * np.sin(thetas)
    zs = np.zeros(xs.size)
    x_err = np.random.normal(scale=noise*radius, size=xs.size)
    y_err = np.random.normal(scale=noise*radius, size=xs.size)
    z_err = np.random.normal(scale=noise*radius, size=xs.size)

    xs += x_err
    ys += y_err
    zs += z_err
    t_stamps = np.arange(n_points)
    ts = t_stamps * t_step
    segment = pd.DataFrame(index=pd.Index(t_stamps, name='t_stamp'),
                           data=np.vstack([xs, ys, zs, x_err, y_err, ts]).T,
                           columns=['x', 'y', 'z', 'x_err', 'y_err', 't'])
    return segment


def test_simple_ellipis_fit():

    segment =  get_mock_segment()
    start = segment.index[0]
    stop = segment.index[-1]
    fit_output, components, rotated = fit_arc_ellipse(segment,
                                                      start, stop,
                                                      ['x', 'y', 'z'],
                                                      return_rotated=True)
    chi2 = np.square(fit_output[2]['fvec']).sum()
    np.testing.assert_almost_equal(chi2, 0, decimal=3)

