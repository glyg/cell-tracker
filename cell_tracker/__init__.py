from .conf import default_metadata
from .conf import ELLIPSIS_CUTOFFS

#from .analysis import Ellipses
from .objects import build_iterator, CellCluster
try:
    from . import ui
except ImportError:
    print('UI tools require IPython and PyQt to function, '
          'resolve those dependencies if you want to use those')
from . import graphics

from .detection import inspect_stack, show_histogram
from .detection import guess_preprocess

from . import data


import logging
log = logging.getLogger(__name__)


__version__ = '0.1-dev'