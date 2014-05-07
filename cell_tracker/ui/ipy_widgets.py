# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import collections

from IPython.html import widgets
from IPython.display import display
from ..conf import default_metadata


def set_metadata(metadata=default_metadata):
    smw = SetMetadataWidget(metadata)
    display(smw)
    for child in smw.children:
        if len(child.children) > 1:
            child.remove_class('vbox')
            child.add_class('hbox')


class SetMetadataWidget(widgets.ContainerWidget):
    ''' IPython widget to handle metadata in a user friendly way
    '''

    def __init__(self, metadata):
        '''
        Creates a widget to handle metadata
        '''

        self.metadata = metadata
        children = []
        for key, val in metadata.items():
            child = TypeAgnosticTextWidget(key, val)
            for grandchild in child.children:
                grandchild.on_trait_change(self.on_value_change, 'value')
            children.append(child)
        self.children = children

    def _update_metadata(self):
        """

        """
        for child in self.children:
            self.metadata[child.description] = child.value

    def on_value_change(self, name, value):
        self._update_metadata()


class TypeAgnosticTextWidget(widgets.ContainerWidget):
    """
    IPython widget that parses the input values and instanciate
    the proper textwidget according to the type.
    """

    def __init__(self, description, value):

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
