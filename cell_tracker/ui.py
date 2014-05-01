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

from IPython.html import widgets
from IPython.display import display, clear_output

from PyQt4 import QtGui

from sktracker.io import StackIO, ObjectsIO
from sktracker.io.utils import load_img_list
from sktracker.trajectories import Trajectories
from . import CellCluster
from .conf import default_metadata


__all__ = ['get_from_excel', 'get_cluster', 'get_name']


def get_from_excel(default_path='.'):
    '''
    This opens a file dialog allowing ot select an excel file containing
    the tracked data, and returns a :class:`CellCluster` object.

    Paramteters
    -----------

    default_path : str
        the path where the file dialog opens

    Returns
    -------

    cellcluster : a :class:`CellCluster` instance
         the container class for the tracking

    Notes
    -----

    The excel file should follow the structure of `excel_trajs_example.xlsx`
    in the project's `data` directory
    '''

    ### Open the file dialog
    data_path, name = get_excel_file(default_path)

    ### Read the data
    trajs = pd.read_excel(data_path, 0)
    trajs.set_index(['t_stamp', 'label'],
                    inplace=True)
    ### The Trajectories class is a subclass of
    ### pandas DataFrame
    trajs = Trajectories(trajs)

    metadata = pd.read_excel(data_path, 1)
    metadata = {name: value for name, value
                in zip(metadata['Name'], metadata['Value'])}

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

def get_cluster(default_path='.', metadata=None):
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

    if metadata is None:
        metadata = default_metadata
    data_path, name = get_dataset(default_path)
    if data_path is None:
        return

    stores = [h5file for h5file in os.listdir(data_path)
              if h5file.endswith('.h5')]
    if len(stores) == 0:
        objectsio = None
    elif len(stores) > 1:
        log.info('''Multiple '.h5' found, loading metadata from images''')
        objectsio = None
    else:
        store_path = os.path.join(data_path, stores[0])
        objectsio = ObjectsIO(store_path=store_path)

    image_path_list = load_img_list(data_path)
    if objectsio is not None:
        stackio = StackIO(image_path_list=image_path_list,
                          metadata=objectsio.metadata)
    else:
        stackio = StackIO(image_path_list=image_path_list,
                          metadata=metadata)
        im0 = stackio.get_tif().asarray()
        if len(im0.shape) == 4:
            stackio.metadata['SizeC'] = im0.shape[0]

    cellcluster = CellCluster(objectsio=objectsio, stackio=stackio )
    return cellcluster

def get_dataset(default='.'):
    '''
    Opens a directory select dialog
    '''
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


def set_metadata(metadata=default_metadata):
    smw = SetMetadataWidget(metadata)
    display(smw)

class SetMetadataWidget(widgets.ContainerWidget):
    ''' IPython widget to handle metadata in a user friendly way
    '''

    def __init__(self, metadata):
        '''
        Creates a widget to handle metadata
        '''

        self.metadata = metadata
        children = []
        for key, val in metadata.items():
            if isinstance(val, np.float):
                child = widgets.FloatTextWidget(description=key,
                                                value=val)
            elif isinstance(val, np.int):
                child = widgets.IntTextWidget(description=key,
                                              value=val)
            elif isinstance(val, str):
                child = widgets.TextWidget(description=key,
                                           value=val)
            child.on_trait_change(self.on_value_change, 'value')
            children.append(child)
        self.children = children

    def _update_metadata(self):
        for child in self.children:
            self.metadata[child.description] = child.value

    def on_value_change(self, name, value):
        self._update_metadata()
