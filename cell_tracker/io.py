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
from sktracker.io.metadataio import METADATA_TYPE
from sktracker.io.utils import load_img_list
from sktracker.trajectories import Trajectories
from .objects import CellCluster
from .conf import default_metadata


def get_cluster(data_path,  metadata=None, single_file=False, extra_sheet=None):
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
        cellcluster = get_from_excel(data_path, extra_sheet)
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


def get_from_excel(data_path, extra_sheet=None):
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
    trajs.t_stamp = trajs.t_stamp.astype(np.int)
    trajs.label = trajs.label.astype(np.int)
    trajs.set_index(['t_stamp', 'label'],
                    inplace=True)

    ### The Trajectories class is a subclass of
    ### pandas DataFrame
    ### Parsing excel files tends to add NaNs to the data
    trajs = Trajectories(trajs.dropna().sortlevel())
    metadata = pd.read_excel(data_path, 1)
    metadata = {name: value for name, value
                in zip(metadata['Name'], metadata['Value'])}

    metadata['FileName'] = data_path
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
    if extra_sheet is not None:
        try:
            extra = pd.read_excel(data_path, extra_sheet)
            cellcluster.extra = extra
            cellcluster.oio['extra'] = extra
        except:
            print('Extra data from sheet {} not found in the file {}'.format(extra_sheet, data_path))
    return cellcluster


def load_multiple_excel(data_path, extra_sheet=None):

    xlsx_file = pd.io.excel.ExcelFile(data_path)

    lastsheet = xlsx_file.book.nsheets - 1
    global_metadata = pd.read_excel(data_path, lastsheet)

    global_metadata = {name: value for name, value
                       in zip(global_metadata['Name'],
                              global_metadata['Value'])}

    clusters = {}
    global_metadata['FileName'] = global_metadata['FileName'].replace(' ', '')
    for i, name in enumerate(global_metadata['FileName'].split(',')):

        ### Read the data
        trajs = pd.read_excel(data_path, i)
        trajs.t_stamp = trajs.t_stamp.astype(np.int)
        trajs.label = trajs.label.astype(np.int)
        trajs.set_index(['t_stamp', 'label'],
                        inplace=True)
        trajs = Trajectories(trajs.dropna())

        metadata = global_metadata.copy()
        metadata['FileName'] = os.path.join(
            os.path.dirname(data_path), name)

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
        clusters[name] = cellcluster

    return clusters