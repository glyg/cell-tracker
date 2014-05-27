# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function



import warnings
warnings.filterwarnings("ignore")
import logging
log = logging.getLogger(__name__)

import sys, os

import pandas as pd
import numpy as np

from PyQt4 import QtGui

from sktracker.io import StackIO, ObjectsIO
from sktracker.io.utils import load_img_list
from sktracker.trajectories import Trajectories
from .. import CellCluster
from ..conf import default_metadata, metadata_types


__all__ = ['get_from_excel', 'get_cluster', 'get_name']


def get_from_excel(data_path):
    '''
    This opens a file dialog allowing ot select an excel file containing
    the tracked data, and returns a :class:`CellCluster` object.

    Paramteters
    -----------

    data_path: the path to the excelTM file

    Returns
    -------

    cellcluster : a :class:`CellCluster` instance
         the container class for the tracking

    Notes
    -----

    The excel file should follow the structure of `excel_trajs_example.xlsx`
    in the project's `data` directory
    '''

    ### Read the data
    trajs = pd.read_excel(data_path, 0)
    trajs.set_index(['t_stamp', 'label'],
                    inplace=True)
    ### The Trajectories class is a subclass of
    ### pandas DataFrame
    ### Parsing excel files tends to add NaNs to the data
    trajs = Trajectories(trajs.dropna())

    metadata = pd.read_excel(data_path, 1)
    metadata = {name: value for name, value
                in zip(metadata['Name'], metadata['Value'])}

    for key, val in metadata.items():
        dtype = metadata_types[key]
        if dtype == tuple:
            tp = val.replace('(', '')
            tp = tp.replace(')', '')
            vs = tp.split(',')
            metadata[key] = tuple(int(v) for v in vs)
        elif dtype == str:
            continue
        else:
            metadata[key] = dtype(val)

    store_path = ''.join(metadata['FileName'].split('.')[:-1]+['.h5'])
    metadata['FileName'] = os.path.join(
        os.path.dirname(data_path), metadata['FileName'])
    store_path = os.path.join(
        os.path.dirname(data_path), store_path)

    ### The ObjectsIO class
    objectsio = ObjectsIO(metadata=metadata, store_path=store_path)
    cellcluster = CellCluster(objectsio=objectsio)
    cellcluster.trajs = trajs
    cellcluster.oio['trajs'] = trajs
    return cellcluster

def get_cluster(default_path='.', metadata=None, single_file=False):
    '''
    This opens a file dialog allowing to select the directory for
    the tracked data, and returns a :class:`CellCluster` object.

    Paramteters
    -----------

    default_path : str
        the path where the file dialog opens

    Returns
    -------

    cellcluster : a :class:`CellCluster` instance
         the container class for the tracking
    '''
    tiff_exts = ['.tiff', '.tif', '.TIFF', '.TIF']
    valid_exts = tiff_exts + ['.h5', '.xlsx']
    filter  = ' '.join(['*{}'.format(ext) for ext in valid_exts])
    objectsio = None
    data_path, name = get_dataset(default_path, filter)
    if data_path is None:
        return

    ## If we find a HDF store, use it
    if data_path.endswith('.h5'):
        store_path = data_path
        objectsio = ObjectsIO(store_path=store_path)
        image_path = objectsio.metadata['FileName']
        image_path_list = None
        if not single_file:
            data_path = os.path.dirname(data_path)
            image_path_list = load_img_list(data_path)

    ## Excel
    elif data_path.endswith('.xlsx'):
        cellcluster = get_from_excel(data_path)
        return cellcluster

    ## Tiff File
    elif any([data_path.endswith(ext) for ext in tiff_exts]):
        if single_file:
            image_path = data_path
            image_path_list = None
        else:
            image_path = None
            data_path = os.path.dirname(data_path)
            image_path_list = load_img_list(data_path)
    if objectsio is not None:
        stackio = StackIO(image_path=image_path,
                          image_path_list=image_path_list,
                          metadata=objectsio.metadata)
    else:
        stackio = StackIO(image_path=image_path,
                          image_path_list=image_path_list,
                          metadata=metadata)
    cellcluster = CellCluster(objectsio=objectsio, stackio=stackio )
    cellcluster.trajs = Trajectories(cellcluster.trajs.dropna())
    return cellcluster

def get_dataset(default='.', filter='*.*'):
    '''
    Opens a directory select dialog
    '''
    app = QtGui.QApplication.instance()
    if not app:
        app = QtGui.QApplication(sys.argv)
    out = QtGui.QFileDialog.getOpenFileName(directory=default, filter=filter)
    if not len(out):
        print('''No data loaded''')
        return None, None

    data_path = str(out)
    splitted = data_path.split(os.path.sep)
    name = '{}_{}'.format(splitted[-2], splitted[-1])

    print('Choosen data path: %s' % data_path)
    return data_path, name

def get_excel_file(default='.'):
    '''
    Opens a file select dialog for xlsx files
    '''
    app = QtGui.QApplication.instance()
    if not app:
        app = QtGui.QApplication(sys.argv)
    out = QtGui.QFileDialog.getOpenFileName(directory=default,
                                            caption='Choose an XLSX file',
                                            filter='*.xlsx')
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
