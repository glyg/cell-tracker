 # -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
import IPython.display as display


import matplotlib
matplotlib.rcParams['backend'] = 'Qt4Agg'
matplotlib.rcParams['savefig.dpi'] = 90
matplotlib.rcParams['image.interpolation'] = 'nearest'
matplotlib.rcParams['image.cmap'] = 'gray'


import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sktracker.trajectories import Trajectories, draw
from sktracker.trajectories.measures import rotation, translation
