 # -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import sys, os
sys.path.append('..')
sys.path.append('../../scikit-tracker')

# Shut up warnings!
import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd

import matplotlib
matplotlib.rcParams['backend'] = 'Qt4Agg'
matplotlib.rcParams['savefig.dpi'] = 90

import matplotlib.pylab as plt

# Various algorithms from other libraries

from mpl_toolkits.mplot3d import Axes3D

from sktracker.io import StackIO, ObjectsIO
from sktracker.io.utils import load_img_list
from sktracker.detection import nuclei_detector
from sktracker.trajectories import Trajectories, draw
import cell_tracker as ct


