import sys, os
sys.path.append('..')
sys.path.append('../../scikit-tracker')

# Shut up warnings!
import warnings
warnings.filterwarnings("ignore")


import numpy as np

import matplotlib
matplotlib.rcParams['backend'] = 'Qt4Agg'
matplotlib.rcParams['savefig.dpi'] = 90

import matplotlib.pylab as plt

import pandas as pd
import json
from PyQt4 import QtGui

from scipy import ndimage

# Various algorithms from other libraries

from mpl_toolkits.mplot3d import Axes3D
from skimage import img_as_float
from skimage import exposure

from scipy.interpolate import splrep, splev
from sklearn.decomposition import PCA

from sktracker.io import StackIO, ObjectsIO
from sktracker.io.utils import load_img_list
from sktracker.detection import nuclei_detector
from sktracker.trajectories import Trajectories, draw
import cell_tracker as ct

import warnings
warnings.filterwarnings("ignore")


def get_cluster(default_path='.', metadata=None):
    if metadata is None:
        metadata = ct.default_metadata

    data_path, name = get_dataset(default_path)
    if data_path is None:
        return
    image_path_list = load_img_list(data_path)
    stackio = StackIO(image_path_list=image_path_list)
    im0 = stackio.get_tif().asarray()

    correct_metadata = {'SizeT': len(image_path_list),
                        'Shape': ((len(image_path_list),)
                                  + im0.shape)}
    if len(im0.shape) == 4:
        correct_metadata['SizeC'] = im0.shape[0]
    stackio.metadata.update(correct_metadata)
    cellcluster = ct.CellCluster(stackio=stackio)
    return cellcluster

def get_dataset(default='.'):
    app = QtGui.QApplication.instance()
    if not app:
        app = QtGui.QApplication(sys.argv)
    out = QtGui.QFileDialog.getExistingDirectory(directory=default)
    if not len(out):
        print('''No data loaded''')
        return None, None

    data_path = str(out)
    splitted = data_path.split(os.path.sep)
    name = '{}_{}'.format(splitted[-2], splitted[-1])

    print('Choosen data path: %s' % data_path)
    return data_path, name


def get_name(default=''):

    app = QtGui.QApplication.instance()
    if not app:
        app = QtGui.QApplication(sys.argv)

    dialog = QtGui.QInputDialog()
    name, ok = dialog.getText(dialog, 'Enter tracker name',
                              'Default is {}:'.format(default))
    if len(name) and ok:
        print('Choosen name: %s' % name)
        return str(name)
    else:
        print('Keeping default name {}'.format(default))
        return default
