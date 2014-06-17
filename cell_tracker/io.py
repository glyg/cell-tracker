# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import sys, os

import pandas as pd
import numpy as np

import logging
log = logging.getLogger(__name__)


from sktracker.io import StackIO, ObjectsIO
from sktracker.io.utils import load_img_list
from sktracker.trajectories import Trajectories
from .objects import CellCluster
from .conf import default_metadata, metadata_types



def get_cluster(data_path,  metadata=None, single_file=False):
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
    objectsio = None

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
    if hasattr(cellcluster, 'trajs'):
        cellcluster.trajs = Trajectories(cellcluster.trajs.dropna())
    return cellcluster


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

    metadata['FileName'] = os.path.join(
        os.path.dirname(data_path), metadata['FileName'])
    store_path = metadata['FileName']
    if '.' in store_path[-6:]:
        store_path = ''.join(store_path.split('.')[:-1]+['.h5'])
    else:
        store_path = store_path+'.h5'
    store_path = os.path.join(
        os.path.dirname(data_path), store_path)

    ### The ObjectsIO class
    objectsio = ObjectsIO(metadata=metadata, store_path=store_path)
    cellcluster = CellCluster(objectsio=objectsio)
    cellcluster.trajs = trajs
    cellcluster.oio['trajs'] = trajs
    return cellcluster
