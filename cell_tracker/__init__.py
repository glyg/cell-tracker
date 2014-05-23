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
from .conf import defaults

detection_parameters = defaults['detection_parameters']
ellipsis_cutoffs = defaults['ellipsis_cutoffs']

import logging
log = logging.getLogger(__name__)


__version__ = '0.1-dev'