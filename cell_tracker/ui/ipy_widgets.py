# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import collections



from IPython.html import widgets
from IPython.display import display
from ..conf import defaults



def set_metadata(metadata=None,
                 to_display=None):

    if metadata is None:
        metadata = defaults['metadata']
    if to_display is None:
        to_display= defaults['metadata_display']

    smw = SettingsWidget(metadata, to_display)
    display(smw)


def set_parameters(paramters=None, to_display=None):
    if paramters is None:
        parameters = defaults['detection_parameters']
    if to_display is None:
        to_display= defaults['detection_display']

    smw = SettingsWidget(parameters, to_display)
    display(smw)


class SettingsWidget(widgets.ContainerWidget):
    ''' IPython widget to handle metadata or parameters in a user friendly way
    '''

    def __init__(self, settings, to_display):
        '''
        Creates a widget to handle settings

        Paramters
        ---------

        settings: a dict like object
            the data to be set
        to_display: a dict like object
            dict or OrderedDict with the keys from `settings`
            that need to be displayed, and the text string to
            appear next to it
        '''
        self.settings = settings
        children = []
        for key in to_display.keys():
            val = self.settings.get(key)
            if val is None:
                continue
            val = self.settings[key]
            descr = to_display[key]
            child = TypeAgnosticTextWidget(key, descr, val)
            for grandchild in child.children:
                grandchild.on_trait_change(self.on_value_change, 'value')
            children.append(child)
        self.children = children

    def _update_settings(self):
        """

        """
        for child in self.children:
            self.settings[child.key] = child.value

    def on_value_change(self, name, value):
        self._update_settings()


class TypeAgnosticTextWidget(widgets.ContainerWidget):
    """
    IPython widget that parses the input values and instanciate
    the proper textwidget according to the type.
    """

    def __init__(self, key, description, value):

        self.key = key
        self.description = description
        if isinstance(value, np.float):
            self.children = (widgets.FloatTextWidget(description=description,
                                                     value=value),)
        elif isinstance(value, np.int):
            self.children = (widgets.IntTextWidget(description=description,
                                                   value=value),)
        elif isinstance(value, str):
            self.children = (widgets.TextWidget(description=description,
                                                value=value),)
        elif isinstance(value, collections.Sequence):
            header = widgets.LatexWidget(value=description)
            display(header)
            # self.children = (header,) + tuple(TypeAgnosticTextWidget(description='', value=sub_val)
            #                                   for sub_val in value)
            self.children = tuple(TypeAgnosticTextWidget(description='', value=sub_val)
                                  for sub_val in value)

    @property
    def value(self):
        if len(self.children) == 1:
            return(self.children[0].value)
        else:
            return([child.value for child in self.children])
