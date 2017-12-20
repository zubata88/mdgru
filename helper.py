__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

TINY = 1e-20
# import nibabel.viewers as vi
from threading import Thread
import matplotlib

matplotlib.use('Agg')
import numpy as np
import copy
import tensorflow as tf
from tensorflow.python.ops import random_ops
import math
import os, errno
import scipy.linalg as la
from scipy.stats import special_ortho_group as sog
from operator import itemgetter
import logging
import urllib.request


def notify_user(chat_id, token, message='no message'):
    """
    Given that a telegram bot has been created, and it's api token is known, a chat_id has been opened, and the
    corresponding chat_id is known, this method can be used to be informed if something happens to our process.
    :param chat_id: chat_id which is used by telegram to communicate with your bot
    :param token: token generated when creating your bot through the BotFather
    :param message: The message to be sent
    """
    try:
        text = urllib.request.urlopen('https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}'
                                      .format(token, chat_id, message)).read()
        logging.getLogger('helper').info('return value of bot message: {}'.format(text))
    except Exception as e:
        logging.getLogger('helper').warning('could not send {} to chat {} of token {}'.format(message, chat_id, token))


def _save_summary_for_2d_image(name, grid, num_channels, collections=[]):
    """
    Helper to summarize 2d images in tensorboard, by saving one for each channel.
    :param name: name of the image to be saved
    :param grid: 2d image
    :param num_channels: num channels to display
    :param collections: which collection should be used for tensorboard, defaults to default summary collection.
    """
    if num_channels == 3 or num_channels == 1:
        tf.summary.image(name, grid, collections=collections)
    else:
        for i, g in enumerate(tf.split(grid, num_channels, axis=-1)):
            tf.summary.image(name + "-c{}".format(i), g, collections=collections)


def save_summary_for_nd_images(name, grid, collections=[]):
    """
    Helper to summarize 3d images in tensorboard, saving an image along each axis.
    :param name: name of image
    :param grid: image data
    :param collections: collection this image is associated with, defaults to the standard tensorboard summary collection
    """
    shape = grid.get_shape().as_list()
    if len(shape) == 4:
        _save_summary_for_2d_image(name, grid, shape[-1], collections)
    elif len(shape) == 5:
        _save_summary_for_2d_image(name + "-d0", grid[:, shape[1] // 2, ...], shape[-1], collections)
        _save_summary_for_2d_image(name + "-d1", grid[:, :, shape[2] // 2, :, :], shape[-1], collections)
        _save_summary_for_2d_image(name + "-d2", grid[..., shape[3] // 2, :], shape[-1], collections)
    else:
        logging.getLogger('helper').warning('we still need to implement image saving for more than 3 dimensions!')


def deprecated(func):
    """
    Decorator function to indicate through our logger that the decorated function should not be used anymore
    :param func: function to decorate
    :return: decorated function
    """
    def print_deprecated(x):
        logging.getLogger('helper').info('this function has been deprecated {}'.format(func))
        func(x)
    return print_deprecated


def convolution_helper_padding_same(inp, filter, filter_shape, strides):
    """
    Helper to allow for convolution strides with less than 1 using transposed convolutions, given that the used padding
    is chosen to be "SAME".
    :param inp:
    :param filter:
    :param filter_shape:
    :param strides:
    :return:
    """
    if not strides or np.min([s for s in strides if s is not None]) >= 1:
        return tf.nn.convolution(inp, filter, "SAME", strides)
    else:
        output_shape = [ii if ii is not None else -1 for ii in inp.get_shape().as_list()]
        output_shape[1:-1] = np.int32(np.round([1/s*o if s is not None else o for s,o in zip(strides,output_shape[1:-1])]))
        output_shape[-1] = filter_shape[-1]
        output_shape[0] = tf.shape(inp)[0]
        filter_shape = copy.copy(filter_shape)
        n = len(filter_shape)
        filter = tf.transpose(filter,[i for i in range(n-2)]+[n-1,n-2])
        filter_shape = filter_shape[:-2]+filter_shape[-2:][::-1]
        if len(filter_shape) < 5:
            op = tf.nn.conv2d_transpose
        elif len(filter_shape) == 5:
            op = tf.nn.conv3d_transpose
        else:
            raise Exception('implement this! should be transposed convolution for the {}d case!'
                            .format(len(filter_shape)-2))
        if len(filter_shape) == 3:
            n_input_shape = [ii if ii is not None else -1 for ii in inp.get_shape().as_list()]
            n_input_shape.insert(1,1)
            old_output_shape = copy.copy(output_shape)
            output_shape.insert(1,1)
            filter_shape.insert(0,1)
            strides = copy.copy(strides)
            strides.insert(0, 1)
            strides = [1]+[int(1/s) if s < 1 else s for s in strides]+[1]
            res = tf.reshape(op(tf.reshape(inp, n_input_shape), tf.reshape(filter, filter_shape), output_shape, strides, "SAME"), old_output_shape)
            return res
        strides = [1] + [int(1/s) if s < 1 else s for s in copy.copy(strides)] + [1]
        return op(inp, filter, output_shape, strides, "SAME")


def force_symlink(file1, file2):
    """
    Tries to create symlink. If it fails, it tries to remove the folder obstructing its way to create one.
    :param file1: path
    :param file2: symlink name
    """
    try:
        os.symlink(file1, file2)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(file2)
            os.symlink(file1, file2)


def argget(dt, key, default=None, keep=False, ifset=None):
    """
    Takes a dictionary dt and a key "key" to evaluate, if this key is contained in the dictionary. If yes, return
    its value. If not, return default. By default, the key and value pair are deleted from the dictionary, except if
    keep is set to True. ifset can be used to override value, and it is returned instead (can obviously not be None).
    :param dt: dictionary to be searched
    :param key: location in dictionary
    :param default: default value if key not found in dictionary
    :param keep: bool, indicating if key shall remain in dictionary
    :param ifset: value override.
    :return: chosen value for key if available. Else default or ifset override.
    """
    if key in dt:

        if keep:
            val = dt[key]
        else:
            val = dt.pop(key)
        if ifset is not None:
            return ifset
        else:
            return val
    else:
        return default


def get_modified_xavier_method(num_elements, uniform_init=False):
    """
    Modified Glorot initializer. returns an initializer using Glorots method for uniform or Gaussian distributions
    depending on the flag "uniform_init".
    :param num_elements: How many elements are there
    :param uniform_init: Shall we use uniform or Gaussian distribution?
    :return: Glorot/Xavier initializer
    """
    if uniform_init:
        def get_modified_xavier(shape, dtype=tf.float32, partition_info=None):
            limit = math.sqrt(3.0 / (num_elements))
            return random_ops.random_uniform(shape, -limit, limit,
                                             dtype, seed=None)
        return get_modified_xavier
    else:
        def get_modified_xavier_normal(shape, dtype=tf.float32, partition_info=None):
            trunc_stddev = math.sqrt(1.3 / (num_elements))
            return random_ops.truncated_normal(shape, 0.0, trunc_stddev, dtype,
                                               seed=None)
        return get_modified_xavier_normal


def np_arr_backward(matrix, n, k1, k2):
    """
    From block block circulant matrix to filter representation using indices
    :param matrix: matrix representation of filter
    :param n: number of channels
    :param k1: filter dim 1
    :param k2: filter dim 2
    :return: filter representation
    """
    return matrix.reshape([n, k1 * k2, n, k1 * k2]).transpose([1, 3, 0, 2])[:, 0, :, :].reshape([k1, k2, n, n])


def np_arr_forward(filt, n, k1, k2):
    """
    From filter to block block circulant matrix representation using indices.
    :param filt: filter variable
    :param n: number of channels
    :param k1: filter dimension 1
    :param k2: filter dimension 2
    :return:  matrix representation of filter
    """
    a, b, c, d, n1, n2 = np.ogrid[0:k1, 0:-k1:-1, 0:k2, 0:-k2:-1, 0:n, 0:n]
    return filt[a + b, c + d, n1, n2].transpose([0, 2, 1, 3, 4, 5]).reshape([k1 * k1, k2 * k2, n, n]).transpose(
        [2, 0, 3, 1]).reshape([k1 * k2 * n, k1 * k2 * n])


def _initializer_Q(n, k1, k2):
    a = np.arange(k1 * k2).reshape([k1, k2, 1, 1])
    bc = np_arr_forward(a, 1, k1, k2)
    to = bc[0, :]
    arr = np.random.random(k1 * k2) - 0.5
    arr = np.asarray([arr[i] if i < to[i] else -arr[to[i]] for i in range(k1 * k2)])
    arr[0] = 0
    skewsymm = np_arr_forward(arr.reshape(k1, k2, 1, 1), 1, k1, k2)

    I = np.eye(k1 * k2)
    return np.float32(np.matmul(la.inv(I + skewsymm), I - skewsymm))


def initializer_W(n, k1, k2):
    Q = _initializer_Q(n, k1, k2)
    if n > 1:
        nxn = sog.rvs(n)
    else:
        return np.float32(Q)
    return np.float32(np.kron(np.float32(nxn), Q))


def get_orthogonal_block_circulant_initialization():
    def get_orthogonal_uniform(shape, dtype=tf.float32, partition_info=None):
        if len(shape) != 4 or shape[2] != shape[3]:
            raise Exception('this is so far only written for 2d convolutions with equal states!')
        return np_arr_backward(initializer_W(shape[2], shape[0], shape[1]), shape[2], shape[0], shape[1])

    return get_orthogonal_uniform


def counter_generator(maxim, pos=0):
    maxim = np.asarray(maxim)
    count = np.zeros(maxim.shape)
    yield copy.deepcopy(count)
    try:
        while True:
            arr = (maxim - count) > 1
            lind = len(arr) - 1 - arr.tolist()[::-1].index(True)
            count[lind] += 1
            count[lind + 1:] = 0
            yield copy.deepcopy(count)
    except ValueError:
        pass
