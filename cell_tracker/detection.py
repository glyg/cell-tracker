import numpy as np
import matplotlib.pylab as plt

from sktracker.detection.nuclei_detector import detect_one_stack


from .graphics import show_histogram
from .conf import detection_parameters


def inspect_stack(cluster, stack_num=0, show=True):

    im0 = cluster.get_z_stack(stack_num)
    shape, maxI, n_uniques = (im0.shape, im0.max(),
                              np.unique(im0).size)

    depth = np.int(np.ceil(np.log2(n_uniques)))
    if depth <= 8:
        cluster.stackio.metadata['SignificantBits'] = 8
        depth = 8
    elif depth <= 12:
        cluster.stackio.metadata['SignificantBits'] = 12
        depth = 12
    elif depth <= 16:
        cluster.stackio.metadata['SignificantBits'] = 16
        depth = 16
    else:
        cluster.stackio.metadata['SignificantBits'] = depth

    print('z stack shape: {}\n'
          'maximum value: {}\n'
          'number of unique values: {}\n'
          'infered signal depth: {}'.format(shape, maxI,
                                            n_uniques, depth))
    if len(shape) > 3:
        print('''Stack doesn't seem to have the correct dimensions, '''
              '''make sure to apply a preprocessing to correct '''
              '''this descrepency''')

    if show:
        fig, (ax0, ax1) = plt.subplots(2, 1)
        proj_im0 = im0
        for ax in range(len(im0.shape[:-2])):
            proj_im0 = proj_im0.max(axis=0)
        ax0.imshow(proj_im0,
                   interpolation='nearest', cmap='gray')
        show_histogram(im0, min(depth, 12), ax1)

    return im0


def single_stack_detection(cluster, stack_num, preprocess):

    metadata = cluster.oio.metadata

    parameters = detection_parameters.copy()
    parameters['min_z_size'] = parameters['min_z_size'] / metadata['PhysicalSizeZ']
    parameters['min_radius'] /= metadata['PhysicalSizeX']
    parameters['max_radius'] /= metadata['PhysicalSizeX']
    parameters['nuc_distance'] /= metadata['PhysicalSizeX']

    stack = cluster.get_z_stack(stack_num)
    if preprocess is not None:
        stack = preprocess(stack)

    one_stack_output = detect_one_stack((stack, parameters),
                                        full_output=True)
    return stack, one_stack_output


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

