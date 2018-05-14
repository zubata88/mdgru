__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

from copy import copy, deepcopy
from helper import argget, convolution_helper_padding_same, compile_arguments
import tensorflow as tf
import numpy as np
from ..crnn.cgru import CGRUCell


class MDRNN(object):

    _defaults = {
        "use_dropconnect_x": True,
        "use_dropconnect_h": True,
        "swap_memory": True,
        "return_cgru_results": False,
        "use_static_rnn": False,
        "no_avg_pool": True,
        "filter_size_x": [7, 7, 7],
        "filter_size_h": [7, 7, 7],
        "crnn_activation": tf.nn.tanh,
        "legacy_cgru_addition": False,
        "crnn_class": CGRUCell,
        "strides": None,

    }

    def __init__(self, inputarr, dropout,
                 dimensions=None, layers=1,
                 num_hidden=100, name="mdgru", **kw):

        '''
        @param inputarr: needs to be in batch, spatialdim1...spatialdimn,channel form
        '''

        mdgru_kw, kw = compile_arguments(self.__class__, transitive=False, **kw)
        for k, v in mdgru_kw.items():
            setattr(self, k, v)
        self.crnn_kw, kw = compile_arguments(self.crnn_class, transitive=True, **kw)

        self.inputarr = inputarr
        self.istraining = argget(kw, 'istraining', tf.constant(True))
        if dimensions is None:
            self.dimensions = [x + 1 for x in range(len(inputarr.get_shape()[1:-1]))]
        else:
            self.dimensions = dimensions
        self.layers = layers
        self.num_hidden = num_hidden
        self.name = name
        self.dropout = dropout

    def __call__(self):
        with tf.variable_scope(self.name):
            outputs = []
            for d in self.dimensions:
                with tf.variable_scope("dim{}".format(d)):
                    fsx = deepcopy(self.filter_size_x)
                    fsh = deepcopy(self.filter_size_h)
                    fsx.pop(d - 1)
                    fsh.pop(d - 1)
                    if self.strides is not None:
                        st = deepcopy(self.strides)
                        stontime = st.pop(d - 1)
                        if self.no_avgpool:
                            def compress_time(inp, fsize, stride, paddtype):
                                chann = inp.get_shape().as_list()[-1]
                                fshape = fsize[1:-1] + [chann, chann]
                                filt = tf.get_variable('compressfilt', fshape)
                                return convolution_helper_padding_same(inp, filt, fshape, stride[1:-1])
                        else:
                            if len(self.strides) == 3:
                                compress_time = tf.nn.avg_pool3d
                            elif len(self.strides) == 2:
                                compress_time = tf.nn.avg_pool
                            else:
                                raise Exception("this wont work avg pool only implemented for 2 and 3d.")
                    else:
                        st = None

                    dimorder = [i for i in range(len(self.inputarr.get_shape()))]
                    dimorder.insert(-1, dimorder.pop(d))
                    dimorder = np.asarray(dimorder)

                    # transpose to correct run direction
                    trans_input = tf.transpose(self.inputarr, dimorder)
                    myshape = [int(x) for x in trans_input.get_shape()[1:]]
                    myshape.insert(0, -1)
                    input_channels = int(trans_input.get_shape()[-1])
                    current_time = int(myshape[-2])
                    tempshape = [-1, current_time, input_channels]

                    # transpose both back:
                    dimorderback = [i for i in range(len(self.inputarr.get_shape()))]
                    dimorderback.insert(d, dimorderback.pop(-2))
                    # forward:
                    with tf.variable_scope('forward'):
                        trans_resf = self.add_cgru(trans_input, myshape, tempshape, fsx=fsx, fsh=fsh, strides=copy(st))
                        if self.strides is not None:
                            ksize = [1 for _ in self.inputarr.get_shape()]
                            ksize[-2] = stontime
                            trans_resf = compress_time(trans_resf, [int(k) if k>=1 else int(np.round(1//k)) for k in ksize], ksize, "SAME")
                        resf = tf.transpose(trans_resf, dimorderback)
                        outputs.append(resf)
                    # backward
                    with tf.variable_scope('backward'):
                        rev_trans_input = tf.reverse(trans_input, axis=[len(dimorder) - 2])
                        # dimorder should now be 1 at location d;)
                        rev_trans_resb = self.add_cgru(rev_trans_input, myshape, tempshape, fsx=fsx, fsh=fsh,
                                                       strides=copy(st))
                        if self.strides is not None:
                            ksize = [1 for _ in self.inputarr.get_shape()]
                            ksize[-2] = stontime
                            rev_trans_resb = compress_time(rev_trans_resb, [int(k) if k>=1 else int(np.round(1//k)) for k in ksize], ksize, "SAME")
                        trans_resb = tf.reverse(rev_trans_resb, axis=[len(dimorder) - 2])
                        resb = tf.transpose(trans_resb, dimorderback)
                        outputs.append(resb)

        if self.return_cgru_results:
            return tf.concat(outputs, len(self.inputarr.get_shape()) - 1)
        else:
            if self.legacy_cgru_addition:
                return tf.add_n(outputs)
            else:
                return tf.add_n(outputs) / len(outputs)

    def add_cgru(self, minput, myshape, tempshape, fsx=[7, 7], fsh=[7, 7], strides=None):
        kw = copy(self.crnn_kw)
        if self.use_dropconnect_h:
            kw["dropconnecth"] = self.dropout
        else:
            kw["dropconnecth"] = None
        if self.use_dropconnect_x:
            kw["dropconnectx"] = self.dropout
        else:
            kw["dropconnectx"] = None
        mycell = self.crnn_class(myshape, self.num_hidden, filter_size_x=fsx, filter_size_h=fsh, strides=strides, **kw)
        trans_input_flattened = tf.reshape(minput, shape=tempshape)
        output_shape = deepcopy(myshape)
        if strides is not None:
            if len(strides) != len(output_shape)-3:
                raise Exception('strides should match the current spatial dimensions')
            output_shape[1:-2] = [int(np.ceil((myshape[1 + i]) / strides[i])) for i in range(len(strides))]
        output_shape[-1] = mycell.output_size

        zeros_dims = tf.stack([tf.shape(minput)[0], np.prod(output_shape[1:-2]), mycell.output_size])
        initial_state = tf.reshape(tf.fill(zeros_dims, 0.0), [-1, mycell.output_size])
        if self.use_static_rnn:
            trans_res_flattened, _ = tf.contrib.rnn.static_rnn(mycell, tf.unstack(trans_input_flattened, axis=-2),
                                                               dtype=tf.float32, initial_state=initial_state)
        else:
            trans_res_flattened, _ = tf.nn.dynamic_rnn(mycell, trans_input_flattened, dtype=tf.float32,
                                                           swap_memory=self.swap_memory, initial_state=initial_state)

        return tf.reshape(trans_res_flattened, shape=output_shape)
