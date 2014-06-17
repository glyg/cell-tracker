import numpy as np
import pandas as pd
from cell_tracker.analysis import fit_arc_ellipse
from cell_tracker import data

def get_mock_ellipsis_segment(radius=30, dtheta=4*np.pi/3,
                              ellipticity=1.5,
                              noise=1e-4, n_points=10,
                              t_step=3, rotation=None):

    a = 2 * radius /(1 + ellipticity)
    b = 2 * radius /(1 + 1/ellipticity)
    t_stamps = np.arange(n_points)
    ts = t_stamps * t_step
    omega = dtheta / ts.max()
    phi_y, x0, y0 = 0, 0, 0
    xs = a * np.cos(omega * ts)
    ys = b * np.sin(omega * ts)
    zs = np.zeros(xs.size)
    x_err = np.random.normal(scale=noise*radius, size=xs.size)
    y_err = np.random.normal(scale=noise*radius, size=xs.size)
    z_err = np.random.normal(scale=noise*radius, size=xs.size)

    xs += x_err
    ys += y_err
    zs += z_err
    segment = pd.DataFrame(index=pd.Index(t_stamps, name='t_stamp'),
                           data=np.vstack([xs, ys, zs, x_err, y_err, ts]).T,
                           columns=['x', 'y', 'z', 'x_err', 'y_err', 't'])
    return segment


def test_simple_ellipis_fit():

    segment =  get_mock_ellipsis_segment()
    start = segment.index[0]
    stop = segment.index[-1]
    fit_data, components, rotated = fit_arc_ellipse(segment,
                                                    start, stop,
                                                    ['x', 'y', 'z'],
                                                    method='polar',
                                                    return_rotated=True)

    assert fit_data is not None
    chi2 = np.exp(-fit_data['gof'])
    np.testing.assert_almost_equal(chi2, 0, decimal=3)


def test_load_from_excel():
    xlsx_file = data.wt_xlsx()

