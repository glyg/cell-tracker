import numpy as np
import matplotlib.pylab as plt

from .graphics import show_histogram

def inspect_stack(stackio, stack_num=0, show_hist=True):

    im0 = stackio.get_tif_from_list(stack_num).asarray()
    shape, maxI, n_uniques = (im0.shape, im0.max(),
                              np.unique(im0).size)

    depth = np.int(np.ceil(np.log2(n_uniques)))
    if depth <= 8:
        stackio.metadata['SignificantBits'] = 8
    elif depth <= 12:
        stackio.metadata['SignificantBits'] = 12
    elif depth <= 16:
        stackio.metadata['SignificantBits'] = 16
    else:
        stackio.metadata['SignificantBits'] = depth

    print('z stack shape: {}\n'
          'maximum value: {}\n'
          'number of unique values: {}\n'
          'infered signal depth: {}'.format(shape, maxI,
                                            n_uniques, depth))
    if show_hist:
        fig, ax = plt.subplots()
        show_histogram(im0, min(depth, 12), ax)
    return im0


def build_iterator(stackio, preprocess=None):

    if preprocess is None:
        iterator = stackio.list_iterator()
    else:
        base_iterator = stackio.list_iterator()
        def iterator():
            for stack in base_iterator():
                yield preprocess(stack)

    return iterator

def guess_preprocess(metadata, max_value, channel=0):
    '''
    At your own risk
    '''
    max_value_depth = np.int(np.ceil(np.log2(max_value)))
    if max_value_depth > metadata['SignificantBits']:
        divider = max_value_depth / metadata['SignificantBits']
    else:
        divider = None
    if 'C' in metadata['DimensionOrder']:
        indexer = (slice(None) if dim is not 'C' else channel
                   for dim in metadata['DimensionOrder'])
        if divider is None:
            def preprocess(stack):
                return stack[indexer]
        else:
            def preprocess(stack):
                return stack[indexer] / divider
    else:
        if divider is None:
            preprocess = None
        else:
            def preprocess(stack):
                return stack / divider
    return preprocess

