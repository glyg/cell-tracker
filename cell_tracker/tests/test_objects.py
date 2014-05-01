# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


from sktracker import data
from sktracker.io import StackIO, ObjectsIO
from cell_tracker import CellCluster



def test_cellcuster_fromoio():

    store_path = data.nuclei_h5_temp()
    oio = ObjectsIO.from_h5(store_path)
    cluster = CellCluster(oio)

