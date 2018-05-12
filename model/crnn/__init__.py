import logging
from copy import deepcopy

import numpy as np
import tensorflow as tf
from tensorflow import sigmoid
from tensorflow.contrib.rnn import LayerRNNCell
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest

from helper import argget, convolution_helper_padding_same, get_modified_xavier_method, \
    get_pseudo_orthogonal_block_circulant_initialization


class CRNNCell(LayerRNNCell):

    def __init__(self, myshape, num_units, activation=tf.nn.tanh, reuse=None, **kw):
        super(CRNNCell, self).__init__(_reuse=reuse)
        self._activation = activation
        self._num_units = num_units
        self.gate = argget(kw, 'gate', sigmoid)
        self.periodicconvolution_x = argget(kw, 'periodicconvolution_x', False)
        self.periodicconvolution_h = argget(kw, 'periodicconvolution_h', False)
        self.filter_size_x = argget(kw, 'filter_size_x', [7, 7])
        self.filter_size_h = argget(kw, 'filter_size_h', [7, 7])
        self.use_bernoulli = argget(kw, 'use_bernoulli', False)
        self.dropconnectx = argget(kw, "dropconnectx", None)
        self.dropconnecth = argget(kw, "dropconnecth", None)
        self.strides = argget(kw, "strides", None)
        if myshape is None:
            raise Exception('myshape cant be None!')
        myshapein = deepcopy(myshape)
        myshapein.pop(-2)
        myshapeout = deepcopy(myshape)
        myshapeout.pop(-2)
        myshapeout[-1] = self._num_units
        if self.strides is not None:
            if len(myshapeout[1:-1]) != len(self.strides):
                raise Exception(
                    'stride shape should match myshapeout[1:-1]! strides: {}, myshape: {}'.format(self.strides,
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
        shape = data.get_shape().as_list()
        # we assume we can drop the channel information from fshape as well as
        # data, thats why we use i+1 (first space reserved for batch) and 
        # ignore last (channel) in data
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

    def _get_dropconnect(self, shape, rate, name):
        if rate is None:
            return None
        if self.use_bernoulli:
            dc = tf.random_uniform(shape, 0, 1, tf.float32, None, name) < rate
            return tf.cast(dc, tf.float32) / rate
        else:
            return tf.random_normal(shape, 1, tf.sqrt((1 - rate) / rate), tf.float32, None, name)

    def _convolution_x(self, data, filterx, filter_shape=None, strides=None):
        if self.periodicconvolution_x:
            # do padding
            data = self._paddata(data, filterx.get_shape().as_list())
            return tf.nn.convolution(data, filterx, "VALID", strides=strides)
        else:
            return convolution_helper_padding_same(data, filterx, filter_shape, strides)

    def _convolution_h(self, data, filterh, filter_shape=None, strides=None):
        if self.periodicconvolution_h:
            # do padding
            data = self._paddata(data, filterh.get_shape().as_list())
            return tf.nn.convolution(data, filterh, "VALID", strides=strides)
        else:
            return convolution_helper_padding_same(data, filterh, filter_shape, strides)

    def _convlinear(self, args, output_size, bias, bias_start=0.0,
                    scope=None, dropconnectx=None, dropconnecth=None, dropconnectxmatrix=None, dropconnecthmatrix=None,
                    strides=None, orthogonal_init=True, **kw):  # dropconnect and dropout are keep probabilities

        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shape = args[1].get_shape().as_list()
        if len(shape) != 2:
            raise ValueError("ConvLinear is expecting 2D arguments: %s" % str(shape))
        if not shape[1]:
            raise ValueError("ConvLinear expects shape[1] of arguments: %s" % str(shape))
        else:
            total_arg_size = shape[1]

        if self.myshapes[1][-1] != total_arg_size:
            logging.getLogger('model').warning('orig_shape does not match.')

        dtype = args[0].dtype

        # Now the computation.
        with vs.variable_scope(scope or "ConvLinear"):
            # reshape to original shape:
            inp = tf.reshape(args[0], self.myshapes[0])
            stat = tf.reshape(args[1], self.myshapes[1])
            # input
            filtershapex = deepcopy(self.filter_size_x)  # [filter_size[0] for _ in range(len(orig_shapes[0][1:-1]))]
            #             strides = [1 for i in filtershape]
            filtershapex.append(self.myshapes[0][-1])
            numelem = np.prod(filtershapex)
            filtershapex.append(output_size)
            filterinp = self._get_weights_x(filtershapex, dtype, numelem, "FilterInp")

            if dropconnectx is not None:
                filterinp *= dropconnectxmatrix

            resinp = self._convolution_x(inp, filterinp, filter_shape=filtershapex, strides=strides)
            # state
            filtershapeh = deepcopy(self.filter_size_h)  # [filter_size[1] for _ in range(len(orig_shapes[1][1:-1]))]
            filtershapeh.append(self.myshapes[1][-1])
            numelem = np.prod(filtershapeh)
            filtershapeh.append(output_size)

            filterstat = self._get_weights_h(filtershapeh, dtype, numelem, "FilterStat",
                                             orthogonal_init=orthogonal_init)
            if dropconnecth is not None:
                filterstat *= dropconnecthmatrix

            resstat = self._convolution_h(stat, filterstat, filter_shape=filtershapeh)

        # back to orig shape
        resinp = tf.reshape(resinp, (-1, output_size))
        resstat = tf.reshape(resstat, (-1, output_size))

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

    def _get_weights_x(self, filtershape, dtype, _numelem, name):
        fs = np.prod(filtershape[:-2])
        num_output = filtershape[-2]
        num_input = filtershape[-1]

        # depending on the activation function, we initialize our weights differently!
        numelem = (fs * num_output + fs * num_input) / 2
        uniform = False
        if self._activation in [tf.nn.elu, tf.nn.relu]:
            numelem = (fs * num_input) / 2
            uniform = False

        return vs.get_variable(
            name, filtershape, dtype=dtype, initializer=get_modified_xavier_method(numelem, uniform))

    def _get_weights_h(self, filtershape, dtype, numelem, name, orthogonal_init=True):
        if len(filtershape) == 4 and orthogonal_init:
            return vs.get_variable(
                name, filtershape, dtype=dtype,
                initializer=get_pseudo_orthogonal_block_circulant_initialization())  # initializer=get_modified_xavier_method(numelem,False))
        else:
            fs = np.prod(filtershape[:-2])
            num_output = filtershape[-2]
            num_input = filtershape[-1]
            numelem = (fs * num_output + fs * num_input) / 2
            uniform = False
            if self._activation in [tf.nn.elu, tf.nn.relu]:
                numelem = (fs * num_input) / 2
                uniform = False

            return vs.get_variable(
                name, filtershape, dtype=dtype, initializer=get_modified_xavier_method(numelem, uniform))
