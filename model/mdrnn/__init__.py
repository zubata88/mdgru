__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import copy
from copy import deepcopy

import numpy as np
import tensorflow as tf

from helper import argget, convolution_helper_padding_same
from helper import get_modified_xavier_method
from model import batch_norm
from ..rnn import CGRU
#not supported anymore, uncomment at your own risk.
#from tensorflow_extra_ops import CaffeBiCGRU3D



class MDGRUNet(object):
    def __init__(self, data, target, dropout, **kw):
        super(MDGRUNet, self).__init__()
        self.bnx = argget(kw, "bnx", False)
        self.bnh = argget(kw, "bnh", False)
        self.bna = argget(kw, "bna", False)
        self.bne = argget(kw, "bne", False)
        self.use_dropconnectx = argget(kw, "use_dropconnectx", True)
        self.use_dropconnecth = argget(kw, "use_dropconnecth", False)
        self.resmdgru = argget(kw, "resmdgru", False)
        self.resgrux = argget(kw, 'resgrux', False)
        self.resgruh = argget(kw, 'resgruh', False)
        self.m = argget(kw, "m", None)
        self.swap_memory = argget(kw, "swap_memory", False)
        self.return_cgru_results = argget(kw, "return_cgru_results", False)
        self.put_r_back = argget(kw, "put_r_back", False)
        self.use_static_rnn = argget(kw, 'use_static_rnn', False)
        self.no_avgpool = argget(kw, 'no_avgpool', True)
        self.filter_sizes = argget(kw, 'filter_sizes', [7, 7, 7])
        self.cgru_activation = argget(kw, 'rnn_activation', tf.nn.tanh)
        self.activation = argget(kw, 'activation', tf.nn.tanh)
        self.use_caffe_impl = argget(kw, "use_caffe_impl", False)
        self.favor_speed_over_memory = argget(kw, "favor_speed_over_memory", True)
        self.use_dropconnect_on_state = argget(kw, 'use_dropconnect_on_state', False)
        self.legacy_cgru_addition = argget(kw, 'legacy_cgru_addition', False)

    def mdgru_bb(self, inp, dropout, num_hidden, num_output, noactivation=False,
                 name=None, **kw):

        dimensions = argget(kw, "dimensions", None)
        if dimensions is None:
            dimensions = [i + 1 for i, v in enumerate(inp.get_shape()[1:-1]) if v > 1]
        bnx = argget(kw, "bnx", self.bnx)
        bnh = argget(kw, "bnh", self.bnh)
        bna = argget(kw, "bna", self.bna)
        bne = argget(kw, "bne", self.bne)
        resmdgru = argget(kw, 'resmdgru', self.resmdgru)
        use_dropconnectx = argget(kw, "use_dropconnectx", self.use_dropconnectx)
        use_dropconnecth = argget(kw, "use_dropconnecth", self.use_dropconnecth)
        cgru_activation = argget(kw, 'cgru_activation', self.cgru_activation)
        myMDGRU = MDGRU
        with tf.variable_scope(name):

            mdgruclass = myMDGRU(inp, dropout, dimensions,
                                 num_hidden=num_hidden,
                                 name="mdgru",
                                 bnx=bnx, bnh=bnh, bna=bna,
                                 use_dropconnectx=use_dropconnectx, use_dropconnecth=use_dropconnecth,
                                 resgrux=self.resgrux,
                                 resgruh=self.resgruh, m=self.m,
                                 return_cgru_results=self.return_cgru_results, swap_memory=self.swap_memory,
                                 put_r_back=self.put_r_back,
                                 cgru_activation=cgru_activation, use_static_rnn=self.use_static_rnn,
                                 no_avgpool=self.no_avgpool,
                                 filter_sizes=self.filter_sizes,
                                 use_dropconnect_on_state=self.use_dropconnect_on_state,
                                 legacy_cgru_addition=self.legacy_cgru_addition,
                                 **kw)
            mdgru = mdgruclass()
            if num_output is not None:
                mdgruinnershape = mdgru.get_shape()[1:-1].as_list()
                doreshape = False
                if len(mdgruinnershape) >= 3:
                    newshape = [-1, np.prod(mdgruinnershape), mdgru.get_shape().as_list()[-1]]
                    mdgru = tf.reshape(mdgru, newshape)
                    doreshape = True
                num_input = mdgru.get_shape().as_list()[-1]
                filtershape = [1 for _ in mdgru.get_shape()[1:-1]] + [num_input, num_output]

                numelem = (num_output + num_input) / 2
                uniform = False
                if self.activation in [tf.nn.elu, tf.nn.relu]:
                    numelem = (num_input) / 2
                    uniform = False
                W = tf.get_variable(
                    "W", filtershape, dtype=tf.float32, initializer=get_modified_xavier_method(numelem, uniform))
                b = tf.get_variable('b', [num_output], initializer=tf.constant_initializer(0))

                mdgru = tf.nn.convolution(mdgru, W, padding="SAME")

                if resmdgru:
                    if doreshape:
                        inp = tf.reshape(inp,
                                         [-1, np.prod(inp.get_shape()[1:-1].as_list()), inp.get_shape().as_list()[-1]])
                    resW = tf.get_variable('resW',
                                           [1 for _ in inp.get_shape().as_list()[1:-1]] + [
                                               inp.get_shape().as_list()[-1], num_output],
                                           dtype=tf.float32, initializer=get_modified_xavier_method(num_output, False))
                    mdgru = tf.nn.convolution(inp, resW, padding="SAME") + mdgru
                if bne:
                    mdgru = batch_norm(mdgru, "bne", mdgruclass.istraining, bias=False, m=mdgruclass.m)
                mdgru = mdgru + b
                if doreshape:
                    mdgru = tf.reshape(mdgru, [-1] + mdgruinnershape + [mdgru.get_shape().as_list()[-1]])
            if noactivation:
                return mdgru
            else:
                return self.activation(mdgru)


class MDGRU(object):
    def __init__(self, inputarr, dropout,
                 dimensions=None, layers=1,
                 num_hidden=100, name="mdgru", **kw):
        '''
        @param inputarr: needs to be in batch, spatialdim1...spatialdimn,channel form
        '''
        self.inputarr = inputarr
        self.add_x_bn = argget(kw, "bnx", False)
        self.add_h_bn = argget(kw, "bnh", False)
        self.add_a_bn = argget(kw, "bna", False)
        self.istraining = argget(kw, 'istraining', tf.constant(True))
        if dimensions is None:
            self.dimensions = [x + 1 for x in range(len(inputarr.get_shape()[1:-1]))]
        else:
            self.dimensions = dimensions
        self.layers = layers
        self.num_hidden = num_hidden
        self.name = name
        self.dropout = dropout
        self.use_dropconnectx = argget(kw, "use_dropconnectx", True)
        self.use_dropconnecth = argget(kw, "use_dropconnecth", False)
        self.use_bernoulli = argget(kw, 'use_bernoulli_dropconnect', False)
        self.m = argget(kw, "min_mini_batch", None)
        self.resgruh = argget(kw, "resgruh", False)
        self.resgrux = argget(kw, "resgrux", False)
        self.filter_sizes = argget(kw, 'filter_sizes', [7, 7, 7])
        self.return_cgru_results = argget(kw, 'return_cgru_results', False)
        self.use_dropconnect_on_state = argget(kw, 'use_dropconnect_on_state', False)
        self.strides = argget(kw, "strides", None)
        self.swap_memory = argget(kw, "swap_memory", True)
        self.put_r_back = argget(kw, "put_r_back", False)
        self.cgru_activation = argget(kw, 'cgru_activation', tf.nn.tanh)
        self.use_static_rnn = argget(kw, 'use_static_rnn', False)
        self.no_avgpool = argget(kw, 'no_avgpool', True)
        self.legacy_cgru_addition = argget(kw, 'legacy_cgru_addition', False)

    def get_all_cgru_outputs(self):
        with tf.variable_scope(self.name):
            outputs = []
            for d in self.dimensions:
                with tf.variable_scope("dim{}".format(d)):
                    fs = copy.deepcopy(self.filter_sizes)
                    fs.pop(d - 1)
                    if self.strides is not None:
                        st = copy.deepcopy(self.strides)
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
                        trans_resf = self.add_cgru(trans_input,
                                                   myshape,
                                                   tempshape,
                                                   dropout=self.dropout,
                                                   m=self.m, fs=fs, strides=copy.copy(st))
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
                        rev_trans_resb = self.add_cgru(rev_trans_input,
                                                       myshape,
                                                       tempshape,
                                                       dropout=self.dropout,
                                                       m=self.m, fs=fs, strides=copy.copy(st))
                        if self.strides is not None:
                            ksize = [1 for _ in self.inputarr.get_shape()]
                            ksize[-2] = stontime
                            rev_trans_resb = compress_time(rev_trans_resb, [int(k) if k>=1 else int(np.round(1//k)) for k in ksize], ksize, "SAME")
                        trans_resb = tf.reverse(rev_trans_resb, axis=[len(dimorder) - 2])
                        resb = tf.transpose(trans_resb, dimorderback)
                        outputs.append(resb)
                        # add both to output vec which will be summed

            return outputs

    def __call__(self):
        if self.return_cgru_results:
            return tf.concat(self.get_all_cgru_outputs(), len(self.inputarr.get_shape()) - 1)
        else:
            outs = self.get_all_cgru_outputs()
            if self.legacy_cgru_addition:
                return tf.add_n(outs)
            else:
                return tf.add_n(outs) / len(outs)

    def add_cgru(self, minput, myshape, tempshape, dropout, m=None, fs=[7, 7],
                 strides=None):
        if self.use_dropconnecth:
            dropconnecth = dropout
        else:
            dropconnecth = None
        if self.use_dropconnectx:
            dropconnectx = dropout
        else:
            dropconnectx = None
        cgruclass = CGRU
        mycell = cgruclass(myshape, self.num_hidden, add_x_bn=self.add_x_bn, add_h_bn=self.add_h_bn, add_a_bn=self.add_a_bn,
                           istraining=self.istraining, m=m, dropconnectx=dropconnectx, dropconnecth=dropconnecth,
                           resgrux=self.resgrux, resgruh=self.resgruh, filter_sizes=fs, strides=strides,
                           put_r_back=self.put_r_back, activation=self.cgru_activation, use_bernoulli=self.use_bernoulli,
                           use_dropconnect_on_state=self.use_dropconnect_on_state,
                           )
        trans_input_flattened = tf.reshape(minput, shape=tempshape)
        padding = np.int32(tempshape[-2])
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


class CaffeMDGRU(object):
    def __init__(self, inputarr, dropout,
                 dimensions=None, layers=1,
                 num_hidden=100, name="mdgru", **kw):
        self.inputarr = inputarr
        self.generator = argget(kw, "generator", False)
        self.add_x_bn = argget(kw, "bnx", False)
        self.add_h_bn = argget(kw, "bnh", False)
        self.add_a_bn = argget(kw, "bna", False)
        self.form = argget(kw, "form", "NDHWC")
        self.istraining = argget(kw, 'istraining', tf.constant(True))
        if dimensions is None:
            self.dimensions = [x + 1 for x in range(len(inputarr.get_shape()[1:-1]))]
        else:
            self.dimensions = dimensions
        self.layers = layers
        self.num_hidden = num_hidden
        self.name = name
        self.dropout = dropout
        self.use_dropconnectx = argget(kw, "use_dropconnectx", True)
        self.use_dropconnecth = argget(kw, "use_dropconnecth", False)
        self.use_bernoulli = argget(kw, 'use_bernoulli_dropconnect', False)
        self.mask_padding = argget(kw, 'maskpadding', None)
        self.m = argget(kw, "min_mini_batch", None)
        self.favor_speed_over_memory = argget(kw, "favor_speed_over_memory", False)
        self.filter_sizes = argget(kw, 'filter_sizes', [7, 7, 7])
        if any([self.add_x_bn, self.add_h_bn, self.add_a_bn]):
            raise Exception("bn not allowed for caffemdgru")

    def __call__(self):
        with tf.variable_scope(self.name):
            outputs = []
            for d in self.dimensions:
                with tf.variable_scope("dim{}".format(d)):
                    outputs.append(self.add_bicgru(d))
            return tf.add_n(outputs)

    def add_bicgru(self, d):
        fs = copy.deepcopy(self.filter_sizes)
        fs.pop(d - 1)
        return CaffeBiCGRU3D(self.inputarr, d - 1, self.num_hidden, self.form,
                             self.dropout if self.use_dropconnectx else None,
                             self.dropout if self.use_dropconnecth else None, fsy=fs[0] // 2, fsz=fs[1] // 2,
                             favorspeedovermemory=self.favor_speed_over_memory,
                             use_bernoulli_dropconnect=self.use_bernoulli)