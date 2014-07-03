 # -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import sys, os
import logging
log = logging.getLogger(__name__)

import numpy as np
import pandas as pd

def continuous_theta(thetas):

    if thetas.shape[0] == 1:
        dthetas = 0
        return dthetas, thetas
    theta0 =  thetas[0]
    dthetas = np.concatenate(([0], np.diff(thetas)))
    dthetas[dthetas > np.pi] -= 2 * np.pi
    dthetas[dthetas < - np.pi] += 2 * np.pi
    thetas = dthetas.cumsum() + theta0
    return dthetas, thetas
