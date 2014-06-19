# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


"""Test dataset and fake auto generated trajectories.

When data function end with _temp. The file is being copied to a temporary
directory before its path is returned.
"""

import os
import tempfile
import shutil
import pandas as pd
import sys

from sktracker.io.utils import load_img_list

data_path = os.path.dirname(os.path.realpath(__file__))


# Image files
def stack_list_dir():
    """
    """
    return os.path.join(data_path, "stack_list")

def stack_list():
    """
    """
    dirname = stack_list_dir()
    file_list = load_img_list(dirname)
    return file_list

# HDF5 files
def nuclei_h5():
    stk_list = stack_list_dir()
    return os.path.join(data_path, stk_list, "Stack-1.h5")


def nuclei_h5_temp():
    """
    """
    d = tempfile.gettempdir()
    f_ori = nuclei_h5()
    f_dest = os.path.join(d, "nuclei.h5")
    shutil.copy(f_ori, f_dest)
    return f_dest

## Excel
def sample_xlsx():
    xlsx_file = os.path.join(data_path,
                            'excel_trajs_example.xlsx')
    return xlsx_file

def wt_xlsx():
    xlsx_file = os.path.join(data_path,
                            'wild_type.xlsx')
    return xlsx_file

def wt_h5():
    h5_file = os.path.join(data_path,
                            'wild_type.h5')
    return h5_file

def armgfp_xlsx():
    xlsx_file = os.path.join(data_path,
                            'ArmGFP.xlsx')
    return xlsx_file

def multiple_xlsx():
    xlsx_file = os.path.join(data_path,
                            'multiple_movies.xlsx')
    return xlsx_file
