import numpy as np
import pandas as pd
from cell_tracker import io
from cell_tracker import data


def test_from_excel():
    xlsx_path = data.wt_xlsx()
    xlsx_name = 'tmp_file'

