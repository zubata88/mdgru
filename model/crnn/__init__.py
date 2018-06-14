__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import logging
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LayerRNNCell
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest

from helper import argget, compile_arguments
from model import convolution_helper_padding_same, get_modified_xavier_method, \
    get_pseudo_orthogonal_block_circulant_initialization


class CRNNCell(LayerRNNCell):
    """Base convolutional RNN method, implements common functions and serves as abstract class.

    Property defaults contains default values for all properties of a CGRUCell that are the same for one MDGRU
    and is used to filter valid arguments.
    :param myshape: Contains shape information on the input tensor.
    :param num_units: Defines number of output channels.
    :param activation: Can be used to override tanh as activation function.
    :param periodic_convolution_x: Enables circular convolution for the input
    :param periodic_convolution_h: Enables circular convolution for the last output / state
    :param dropconnectx: Enables dropconnect regularization on weights connecting to input
    :param dropconnecth: Enables dropconnect regularization on weights connecting to previous state / output
    """

    _defaults = {
        "periodic_convolution_x": False,
        "periodic_convolution_h": False,
        "use_bernoulli": False,
        "dropconnectx": None,
        "dropconnecth": None,
        "crnn_activation": tf.nn.tanh,
    }

    def _default_crnn_activation(self):
        return tf.nn.tanh

    def __init__(self, myshape, num_units, kw):
        super(CRNNCell, self).__init__()
        crnn_kw, kw = compile_arguments(CRNNCell, kw, transitive=False)
        for k, v in crnn_kw.items():
            setattr(self, k, v)
        self._num_units = num_units
        self.filter_size_x = argget(kw, "filter_size_x", [7, 7])
        self.filter_size_h = argget(kw, "filter_size_h", [7, 7])
        self.strides = argget(kw, "strides", None)
        if myshape is None:
            raise Exception("myshape cant be None!")
        myshapein = deepcopy(myshape)
        myshapein.pop(-2)
        myshapeout = deepcopy(myshape)
        myshapeout.pop(-2)
        myshapeout[-1] = self._num_units
        if self.strides is not None:
            if len(myshapeout[1:-1]) != len(self.strides):
                raise Exception(
                    "stride shape should match myshapeout[1:-1]! strides: {}, myshape: {}".format(self.strides,
                                                                                                  myshapeout))
            myshapeout[1:-1] = [int(np.round((myshapeout[1 + si]) / self.strides[si])) for si in
                                range(len(myshapeout) - 2)]
        self.myshapes = (myshapein, myshapeout)

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def _paddata(self, data, fshape):
        """Pads spatial dimensions of data, such that a convolution of size fshape results in a circular convolution"""
        shape = data.get_shape().as_list()
        for i, j in enumerate(fshape[:-2]):
            begin1 = np.zeros(len(shape), dtype=np.int32)
            size1 = -np.ones(len(shape), dtype=np.int32)
            begin2 = np.zeros(len(shape), dtype=np.int32)
            size2 = -np.ones(len(shape), dtype=np.int32)
            front = (j - 1) // 2
            back = j - 1 - front
            begin1[i + 1] = shape[i + 1] - front
            size1[i + 1] = front
            size2[i + 1] = back
            data = tf.concat([tf.slice(data, begin1, size1), data, tf.slice(data, begin2, size2)], i + 1)
        return data

    def _get_dropconnect(self, shape, keep_rate, name):
        """Creates factors to be applied to filters to achieve either Bernoulli or Gaussian dropconnect."""
        if keep_rate is None:
            return None
        if self.use_bernoulli:
            dc = tf.random_uniform(shape, 0, 1, tf.float32, None, name) < keep_rate
            return tf.cast(dc, tf.float32) / keep_rate
        else:
            return tf.random_normal(shape, 1, tf.sqrt((1 - keep_rate) / keep_rate), tf.float32, None, name)

    def _convolution(self, data, convolution_filter, filter_shape=None, strides=None, is_circular_convolution=False):
        """Convolves data and convolution_filter, using circular convolution if required."""
        if is_circular_convolution:
            data = self._paddata(data, convolution_filter.get_shape().as_list())
            return tf.nn.convolution(data, convolution_filter, "VALID", strides=strides)
        else:
            return convolution_helper_padding_same(data, convolution_filter, filter_shape, strides)

    def _convlinear(self, args, output_size, bias, bias_start=0.0,
                    scope=None, dropconnectx=None, dropconnecth=None, dropconnectxmatrix=None, dropconnecthmatrix=None,
                    strides=None, orthogonal_init=True):
        """Computes the convolution of current input and previous output or state (args[0] and args[1]).

        The two tensors contained in args are convolved with their respective filters. Due to the rnn library of
        tensorflow, spatial dimensions are collapsed and have to be restored before convolution. Also,
        dropconnectmatrices are applied to the weights. If specified, a bias is generated and returned as well.
        :param args: Current input and last output in a list
        :param output_size: Number of output channels (separate from myshapes[1][-1], as sometimes this value differs)
        :param bias: Flag if bias should be used
        :param bias_start: Flag for bias initialization
        :param scope: Override standard "ConvLinear" scope
        :param dropconnectx: Flag if dropconnect should be applied on input weights
        :param dropconnecth: Flag if dropconnect should be applied on state weights
        :param dropconnectxmatrix: Dropconnect matrix for input weights
        :param dropconnecthmatrix: Dropconnect matrix for state weights
        :param strides: Strides to be applied to the input convolution
        :param orthogonal_init: Flag if orthogonal initialization should be performed for the state weights
        :return: 2-tuple of results for state and input, 3-tuple additionally including a bias if requested
        """
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        # Calculate the total size of arguments on dimension 1.
        shape = args[1].get_shape().as_list()
        if len(shape) != 2:
            raise ValueError("ConvLinear is expecting 2D arguments: %s" % str(shape))
        if not shape[1]:
            raise ValueError("ConvLinear expects shape[1] of arguments: %s" % str(shape))
        if self.myshapes[1][-1] != shape[1]:
            logging.getLogger("model").warning("orig_shape does not match.")
        dtype = args[0].dtype
        # Now the computation.
        with vs.variable_scope(scope or "ConvLinear"):
            # Reshape to original shape:
            inp = tf.reshape(args[0], self.myshapes[0])
            stat = tf.reshape(args[1], self.myshapes[1])
            # Prepare convolution filter for the input.
            filtershapex = deepcopy(self.filter_size_x)
            filtershapex.append(self.myshapes[0][-1])
            filtershapex.append(output_size)
            filterinp = self._get_weights_x(filtershapex, dtype, "FilterInp")
            # Regularize input weights.
            if dropconnectx is not None:
                filterinp *= dropconnectxmatrix
            # Convolve input.
            resinp = self._convolution(inp, filterinp, filter_shape=filtershapex, strides=strides,
                                       is_circular_convolution=self.periodic_convolution_x)
            # Prepare convolution filter for the state.
            filtershapeh = deepcopy(self.filter_size_h)  # [filter_size[1] for _ in range(len(orig_shapes[1][1:-1]))]
            filtershapeh.append(self.myshapes[1][-1])
            filtershapeh.append(output_size)
            filterstat = self._get_weights_h(filtershapeh, dtype, "FilterStat", orthogonal_init=orthogonal_init)
            # Regularize state weights.
            if dropconnecth is not None:
                filterstat *= dropconnecthmatrix
            # Convolve state.
            resstat = self._convolution(stat, filterstat, filter_shape=filtershapeh,
                                        is_circular_convolution=self.periodic_convolution_h)
        # Back to original shape.
        resinp = tf.reshape(resinp, (-1, output_size))
        resstat = tf.reshape(resstat, (-1, output_size))
        # Add and return bias if flag is set, otherwise return above results only.
        if bias:
            bias_term = vs.get_variable(
                "Bias", [output_size],
                dtype=dtype,
                initializer=init_ops.constant_initializer(
                    bias_start, dtype=dtype))
            resbias = tf.reshape(bias_term, (-1, output_size))
            return resinp, resstat, resbias
        else:
            return resinp, resstat

    def _get_weights_x(self, filtershape, dtype, name):
        """Return weights for input convolution."""
        fs = np.prod(filtershape[:-2])
        num_output = filtershape[-2]
        num_input = filtershape[-1]
        # depending on the activation function, we initialize our weights differently!
        numelem = (fs * num_output + fs * num_input) / 2
        uniform = False
        if self.crnn_activation in [tf.nn.elu, tf.nn.relu]:
            numelem = (fs * num_input) / 2
            uniform = False
        return vs.get_variable(
            name, filtershape, dtype=dtype, initializer=get_modified_xavier_method(numelem, uniform))

    def _get_weights_h(self, filtershape, dtype, name, orthogonal_init=True):
        """Return weights for output convolution."""
        if len(filtershape) == 4 and orthogonal_init:
            return vs.get_variable(
                name, filtershape, dtype=dtype,
                initializer=get_pseudo_orthogonal_block_circulant_initialization())
        else:
            fs = np.prod(filtershape[:-2])
            num_output = filtershape[-2]
            num_input = filtershape[-1]
            numelem = (fs * num_output + fs * num_input) / 2
            uniform = False
            if self.crnn_activation in [tf.nn.elu, tf.nn.relu]:
                numelem = (fs * num_input) / 2
                uniform = False
            return vs.get_variable(
                name, filtershape, dtype=dtype, initializer=get_modified_xavier_method(numelem, uniform))
