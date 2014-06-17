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

from PyQt4 import QtGui
from .. import io

def get_cluster(default_path):

    tiff_exts = ['.tiff', '.tif', '.TIFF', '.TIF']
    valid_exts = tiff_exts + ['.h5', '.xlsx']
    ext_filter  = ' '.join(['*{}'.format(ext) for ext in valid_exts])

    data_path, name = get_dataset(default_path, ext_filter)
    if data_path is None:
        return
    cellcluster = io.get_cluster(data_path)
    return cellcluster


def get_dataset(default='.', ext_filter='*.*'):
    '''
    Opens a directory select dialog
    '''
    app = QtGui.QApplication.instance()
    if not app:
        app = QtGui.QApplication(sys.argv)
    out = QtGui.QFileDialog.getOpenFileName(directory=default, filter=ext_filter)
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
