import numpy as np
from collections import OrderedDict


defaults = {}

default_metadata =  {"PhysicalSizeX": 0.42,
                     "PhysicalSizeY": 0.42,
                     "PhysicalSizeZ": 1.5,
                     "TimeIncrement": 3,
                     "FileName": '',
                     "Shape": [1, 1, 512, 512],
                     "DimensionOrder": "TZYX" }

metadata_types = {"PhysicalSizeX": np.float,
                  "PhysicalSizeY": np.float,
                  "PhysicalSizeZ": np.float,
                  "TimeIncrement": np.float,
                  "FileName": str,
                  "Shape": tuple,
                  "DimensionOrder": str,
                  'SizeX': np.int,
                  'SizeY': np.int,
                  'SizeZ': np.int,
                  'SizeC': np.int,
                  'SizeT': np.int}

defaults['metadata'] = default_metadata

metadata_display = OrderedDict([('PhysicalSizeX',
                                 'X pixel size'),
                                ('PhysicalSizeY',
                                 'Y pixel size'),
                                ('SizeT',
                                 'Number of time points'),
                                ('PhysicalSizeZ',
                                 'Z step'),
                                ('SizeZ',
                                 'Number of Z planes per time point'),
                                ('TimeIncrement',
                                 'Time step between two stacks')])

defaults['metadata_display'] = metadata_display


detection_parameters = {'segment_method': 'otsu',
                        'correction': 1.,
                        'smooth': 10,
                        'min_radius': 2.,
                        'max_radius': 8.,
                        'num_cells': 8,
                        'nuc_distance':6,
                        'min_z_size': 4}

defaults['detection_parameters'] = detection_parameters

detection_display = OrderedDict([('segment_method',
                                  "Segmentation method ('otsu' or 'naive')"),
                                 ('correction',
                                  'Threshold correction (between 0 and 1)'),
                                 ('smooth',
                                  'Smoothing size in pixels'),
                                 ('min_radius',
                                  'Minimum radius'),
                                 ('max_radius',
                                  'Maximum radius'),
                                 ('nuc_distance',
                                  'Minimum distance between two nuclei'),
                                 ('min_z_size',
                                  'Minimum size along the Z axis')])

defaults['detection_display'] = detection_display

ellipsis_cutoffs = {'max_ellipticity': 3.,
                    'min_gof': 1.,
                    'max_dtheta': 720.,
                    'min_dtheta': 30.,
                    'max_radius': 60.,
                    'min_radius': 5.}

defaults['ellipsis_cutoffs'] = ellipsis_cutoffs

ellipsis_display = OrderedDict([('max_ellipticity',
                                 'Maximum ellipticity (long axis / short axis)'),
                                ('min_gof',
                                 'Minimal goodness of fit (-log(chi**2))'),
                                ('max_dtheta',
                                 'Maximum angular amplitude (degrees)'),
                                ('min_dtheta',
                                 'Minimum angular amplitude (degrees)'),
                                ('max_radius',
                                 'Maximum ellipsis radius (µm)'),
                                ('min_radius',
                                 'Minimum ellipsis radius (µm)'),
                            ])

defaults['ellipsis_display'] = ellipsis_display

