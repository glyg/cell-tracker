# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from skimage import img_as_float, filter

'''
Collection of simple function to apply to a stack
before passing it to the detector.

All those functions should take a stack as input
and return a stack.
'''

all = ['no_preprocess', 'normalize',
       'highpass', 'from16to8',
       'red_channel', 'blue_channel',
       'green_channel', 'red_16to8']

no_preprocess = None

def normalize(stack):
    '''Returns a stack with maximum value at one,
    (and thus broadcasted to float data type)
    '''
    return stack / stack.max()

def highpass(stack, smth_width=100):
    '''
    Simulates a high pass filter in Fourier space by
    substracting a smoothed version of the input image.
    '''
    stack = img_as_float(stack/stack.max())
    lowpass = filter.gaussian_filter(stack, smth_width)
    f_stack = stack - lowpass
    f_stack -= f_stack.min()
    return f_stack

def from16to8(stack):
    '''
    Brutally devides input stack
    by 2**8 (necessited for strangely
    converted images)
    '''
    out = stack // 256
    return out.astype(np.uint8)

def red_channel(stack):
    return stack[:, 0, :, :]

def green_channel(stack):
    return stack[:, 1, :, :]

def blue_channel(stack):
    return stack[:, 2, :, :]


def red_16to8(stack):
    '''Brutally devides input stack ZCXY
    by 2**8 and returns only channel C=0.

    Usefull for strangely converted confocal stacks.
    '''
    out = stack[:, 0, :, :] // 256
    return out.astype(np.uint8)
