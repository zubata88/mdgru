__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import copy
from copy import deepcopy

import numpy as np
import tensorflow as tf

from helper import argget, convolution_helper_padding_same
from helper import get_modified_xavier_method
from model import batch_norm
# CaffeMDGRU is not supported anymore, uncomment at own risk:
from .caffe_mdgru import CaffeMDGRU
from .mdgru import MDGRU


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


