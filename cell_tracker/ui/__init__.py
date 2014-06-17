try :
    from .ipy_widgets import set_metadata
    from .ipy_widgets import set_parameters
    from .ipy_widgets import SettingsWidget
    from .qt_dialogs import get_cluster
    from .manual_tracking import ManualTracking, pick_border_cells
except ImportError:
    print('UI elements could not be loaded, try installing IPython and PyQt4')
