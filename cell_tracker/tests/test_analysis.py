import numpy as np
import pandas as pd
from cell_tracker.analysis import arc_ellipse


def get_mock_segment(radius=30, dtheta=4*np.pi/3,
                     ellipticity=1.5, noise=1e-4, t_span=90, t_step=3):

    a = radius * ellipticity
    b = radius / ellipticity

    omega = dtheta/t_span

    phi_x, phi_y, x0, y0 = 0, 0, 0, 0

    params = a, b, omega, phi_x, phi_y, x0, y0
    ts = np.arange(0, t_span, t_step)
    t_stamps = ts / t_step
    t_stamps = t_stamps.astype(np.int)

    x, y = arc_ellipse(ts, params)
    z = np.ones(x.size)
    x_err = np.random.normal(scale=noise*radius, size=x.size)
    y_err = np.random.normal(scale=noise*radius, size=x.size)
    x += x_err
    y += y_err

    segment = pd.DataFrame(index=pd.Index(t_stamps, name='t_stamp'),
                           data=np.vstack([x, y, z, x_err, y_err, ts]).T,
                           columns=['x', 'y', 'z', 'x_err', 'y_err', 't'])
    return segment
