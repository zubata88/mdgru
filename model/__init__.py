__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import tensorflow as tf
import numpy as np
from helper import argget, compile_arguments
import copy
from tensorflow.python import pywrap_tensorflow
import logging
from tensorflow.python.ops import random_ops
import math
from helper import np_arr_backward, initializer_W, lazy_property

def _save_summary_for_2d_image(name, grid, num_channels, collections=[]):
    """ Helper to summarize 2d images in tensorboard, by saving one for each channel.
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
    """ Helper to summarize 3d images in tensorboard, saving an image along each axis.
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
        logging.getLogger('helper').warning('Saving images with more than 3 dimensions in Tensorboard is not '
                                            'implemented!')


def get_pseudo_orthogonal_block_circulant_initialization():
    """ Creates pseudo-orthogonal initialization for given shape.

    Pseudo-orthogonal initialization is achieved assuming circular convolution and a signal size equal to the filter
    size. Hence, if applied to signals larger than the filter, or not using circular convolution leads to non orthogonal
    filter initializations.

    :return: pseudo-orthogonal initialization for given shape
    """

    def get_pseudo_orthogonal_uniform(shape, dtype=tf.float32, partition_info=None):
        if len(shape) != 4 or shape[2] != shape[3]:
            raise Exception('this is so far only written for 2d convolutions with equal states!')
        return np_arr_backward(initializer_W(shape[2], shape[0], shape[1]), shape[2], shape[0], shape[1])

    return get_pseudo_orthogonal_uniform


def get_modified_xavier_method(num_elements, uniform_init=False):
    """ Modified Glorot initializer.

    Returns an initializer using Glorots method for uniform or Gaussian distributions
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


def convolution_helper_padding_same(inp, convolution_filter, filter_shape, strides):
    """ Helper to allow for convolution strides with less than 1.

    Strides less than one are performed using transposed convolutions, strides larger than one are performed using
    normal convolutions. This helper function only works if all filters are larger or equal than one or smaller or equal
    than 1. If this is not given, an error is raised. given that the used padding
    is chosen to be "SAME".
    :param inp: input data to convolve
    :param convolution_filter: filter to perform convolution with
    :param filter_shape: filter shape as list
    :param strides: list of strides to be used for the spatial dimensions of inp during the convolution.
    :return:
    """
    if not strides or len([s for s in strides if s is not None]) == 0 \
            or np.min([s for s in strides if s is not None]) >= 1:
        return tf.nn.convolution(inp, convolution_filter, "SAME", strides)
    else:
        if np.max([s for s in strides if s is not None]) > 1:
            raise Exception('Mixes of strides above and below 1 for one convolution operation are not supported!')
        output_shape = [ii if ii is not None else -1 for ii in inp.get_shape().as_list()]
        output_shape[1:-1] = np.int32(
            np.round([1 / s * o if s is not None else o for s, o in zip(strides, output_shape[1:-1])]))
        output_shape[-1] = filter_shape[-1] # get number of output channel
        output_shape[0] = tf.shape(inp)[0] # get batchsize
        filter_shape = copy.copy(filter_shape)
        n = len(filter_shape)
        # switch last two dimensions of convolution filter and adjust filter_shape:
        convolution_filter = tf.transpose(convolution_filter, [i for i in range(n - 2)] + [n - 1, n - 2])
        filter_shape = filter_shape[:-2] + filter_shape[-2:][::-1]
        # select up/transposed convolution operation
        if len(filter_shape) < 5:
            op = tf.nn.conv2d_transpose
        elif len(filter_shape) == 5:
            op = tf.nn.conv3d_transpose
        else:
            raise Exception('Transposed convolution is not implemented for the {}d case!'
                            .format(len(filter_shape) - 2))
        # special preparation to use conv2d_transpose for 1d upconvolution:
        if len(filter_shape) == 3:
            old_output_shape = copy.copy(output_shape)
            n_input_shape = [ii if ii is not None else -1 for ii in inp.get_shape().as_list()]
            # add singleton dimension for each tensor
            n_input_shape.insert(1, 1)
            output_shape.insert(1, 1)
            filter_shape.insert(0, 1)
            strides = copy.copy(strides)
            strides.insert(0, 1)
            strides = [1] + [int(1 / s) if s < 1 else s for s in strides] + [1]
            res = tf.reshape(
                op(tf.reshape(inp, n_input_shape), tf.reshape(convolution_filter, filter_shape), output_shape, strides,
                   "SAME"), old_output_shape)
            return res
        strides = [1] + [int(1 / s) if s < 1 else s for s in copy.copy(strides)] + [1]
        return op(inp, convolution_filter, output_shape, strides, "SAME")


def batch_norm(x, name_scope, training, epsilon=1e-3, decay=0.999, bias=True, m=None):
    """Computes the batch_norm for x"""
    with tf.variable_scope(name_scope):
        size = x.get_shape().as_list()[-1]

        scale = tf.get_variable("scale", [size], initializer=tf.constant_initializer(0.1))
        if bias:
            offset = tf.get_variable("offset", [size])
        else:
            offset = None

        pop_mean = tf.get_variable("pop_mean", [size], initializer=tf.zeros_initializer(), trainable=False)
        pop_var = tf.get_variable("pop_var", [size], initializer=tf.constant_initializer(1.0), trainable=False)
        batch_mean, batch_var = tf.nn.moments(x, [i for i in range(len(x.get_shape()) - 1)])

        # The following simulates a mini-batch for scenarios where we don't have
        # a large enough mini-batch (THIS ONLY WORKS FOR BATCH SIZES OF 1)
        if m is not None:
            batch_mean_list = tf.get_variable("batch_mean_list", [m, size], initializer=tf.zeros_initializer(),
                                              trainable=False)
            batch_var_list = tf.get_variable("batch_var_list", [m, size], initializer=tf.constant_initializer(1.0),
                                             trainable=False)
            starter = tf.get_variable("batch_list_starter", initializer=tf.constant(0.0), trainable=False,
                                      dtype=tf.float32)

            def alter_list_at(data, counter, line):
                data_unpacked = tf.unstack(data)
                data_unpacked.pop(0)
                data_unpacked.append(line)
                return tf.stack(data_unpacked)

            starter_op = tf.assign(starter, starter + 1)
            with tf.control_dependencies([starter_op]):
                batch_mean_list_op = tf.assign(batch_mean_list, alter_list_at(batch_mean_list, starter, batch_mean))
                batch_var_list_op = tf.assign(batch_var_list, alter_list_at(batch_var_list, starter, batch_var))
            with tf.control_dependencies([batch_mean_list_op, batch_var_list_op]):
                batch_mean = tf.reduce_sum(batch_mean_list, 0) / tf.minimum(starter, m)
                batch_var = tf.reduce_sum(batch_var_list, 0) / tf.minimum(starter, m - 1)

        def batch_statistics():
            train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)

        def population_statistics():
            return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

        return tf.cond(training, batch_statistics, population_statistics)


class Model(object):
    """Abstract Model class"""
    _defaults = {
        "model_seed": {'value': 12345678, 'help': "Override default model initialization random seed"},
    }

    def __init__(self, data, target, dropout, kw):
        self.origargs = copy.copy(kw)
        print("model")
        model_kw, kw = compile_arguments(Model, kw, transitive=False)
        for k, v in model_kw.items():
            setattr(self, k, v)

        # self.model_seed = argget(kw, 'model_seed', 12345678)
        tf.set_random_seed(self.model_seed)
        super(Model, self).__init__(data, target, dropout, kw)
        self.training = argget(kw, "training", tf.constant(True))
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.use_tensorboard = argget(kw, "use_tensorboard", True)
        #
        # if argget(kw, "whiten", False):
        #     self.data = batch_norm(data, "bn", self.training, m=32)
        # else:
        self.data = data
        # pass
        self.dimensions = argget(kw, "dimensions", None)

    def prediction(self):
        """lazy property call to produce a prediction in tensorflow."""
        raise Exception("this should never be called, but implemented by"
                        "the child class")

    def costs(self):
        """lazy property to compute the costs per sample."""
        raise Exception("this should never be called, but implemented by"
                        "the child class")

    @lazy_property
    def cost(self):
        """lazy property to compute the cost per batch"""
        loss = tf.reduce_mean(self.costs)
        if self.use_tensorboard:
            tf.summary.scalar("loss", loss)
        return loss

    @staticmethod
    def get_model_name_from_ckpt(ckpt):
        """returns root node name of tensorflow graph stored in checkpoint ckpt"""
        try:
            r = pywrap_tensorflow.NewCheckpointReader(ckpt)
            modelname = r.get_variable_to_shape_map().popitem()[0].split('/')[0]
        except:
            logging.getLogger('runfile').warning('could not load modelname from ckpt-file {}'.format(ckpt))
            modelname = None
        return modelname

    @staticmethod
    def set_allowed_gpu_memory_fraction(gpuboundfraction):
        tf.GPUOptions(per_process_gpu_memory_fraction=gpuboundfraction)


class ClassificationModel(Model):
    """Abstract model class. """
    def __init__(self, data, target, dropout, kw):
        print("classificationmodel")
        super(ClassificationModel, self).__init__(data, target, dropout, kw)
        self.target = target
        self.dropout = dropout
        self.learning_rate = argget(kw, "learning_rate", 1)
        self.momentum = argget(kw, "momentum", 0.9)
        self.nclasses = argget(kw, "nclasses", 2)


class RegressionModel(Model):
    """Abstract model class for regression tasks."""
    def __init__(self, data, target, dropout, kw):
        super(RegressionModel, self).__init__(data, target, dropout, kw)
        self.target = target
        self.dropout = dropout
        self.learning_rate = argget(kw, "learning_rate", 0.001)
        self.nclasses = argget(kw, "nclasses", 1)
        self.momentum = argget(kw, "momentum", 0.9)


class ReconstructionModel(Model):
    """Abstract model class for reconstruction tasks."""
    def __init__(self, data, dropout, kw):
        super(ReconstructionModel, self).__init__(data, dropout, None, kw)
        self.dropout = dropout
        self.learning_rate = argget(kw, "learning_rate", 0.001)
        self.nclasses = argget(kw, "nclasses", 2)


class GANModel(Model):
    """Abstract model class for GANs."""
    def __init__(self, data, dropout, kw):
        super(GANModel, self).__init__(data, dropout, None, kw)
        self.dropout = dropout
        self.learning_rate = argget(kw, "learning_rate", 0.001)
        self.momentum = argget(kw, "momentum", 0.9)
        self.nclasses = argget(kw, "nclasses", 2)
        self.fakedata = argget(kw, "fakedata", None)
