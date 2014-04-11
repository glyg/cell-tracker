import numpy as np

default_metadata =  {"PhysicalSizeX": 0.42,
                     "PhysicalSizeY": 0.42,
                     "PhysicalSizeZ": 1.5,
                     "TimeIncrement": 3,
                     "FileName": '',
                     "Shape": [1, 1, 512, 512],
                     "DimensionOrder": "TZYX" }

ELLIPSIS_CUTOFFS = {'max_ellipticity': 3.,
                    'min_gof': 1.,
                    'max_dtheta': 4 * np.pi,
                    'min_dtheta': np.pi / 6.,
                    'max_radius': 60,
                    'min_radius': 5}
