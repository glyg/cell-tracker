import numpy as np
import pandas as pd
from cell_tracker import io
from cell_tracker import data


def test_from_excel():
    xlsx_path = data.wt_xlsx()
    cluster = io.get_cluster(xlsx_path)
    assert cluster.trajs.shape == (175, 4)

def test_from_h5():
    h5_path = data.wt_h5()
    cluster = io.get_cluster(h5_path)
    assert cluster.trajs.shape == (175, 19)

def test_multi_excel():
    xlsx_path = data.multiple_xlsx()
    clusters = io.load_multiple_excel(xlsx_path)
    assert len(clusters.keys()) == 3
