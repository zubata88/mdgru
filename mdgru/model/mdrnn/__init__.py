__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import numpy as np
import tensorflow as tf

from mdgru.helper import argget, generate_defaults_info
from mdgru.helper import compile_arguments
from mdgru.model import batch_norm, get_modified_xavier_method
# CaffeMDGRU is not supported anymore, uncomment at own risk:
# from .caffe_mdgru import CaffeMDGRU
from mdgru.model.mdrnn.mdgru import MDRNN


class MDGRUNet(object):
    """Convenience class combining attributes to be used for multiple MDRNN and voxel-wise fully connected layers."""
    _defaults = {
        "add_e_bn": False,
        "resmdgru": {'value': False, 'help': 'Add a residual connection around an MDGRU block'},
        "vwfc_activation": tf.nn.tanh,
    }

    def __init__(self, data, target, dropout, kw):
        super(MDGRUNet, self).__init__()
        mdrnn_net_kw, kw = compile_arguments(MDGRUNet, kw, transitive=False)
        for k, v in mdrnn_net_kw.items():
            setattr(self, k, v)
        self.mdrnn_kw, kw = compile_arguments(MDRNN, kw, transitive=True)
        self.crnn_kw, kw = compile_arguments(self.mdrnn_kw['crnn_class'], kw, transitive=True)

    def mdgru_bb(self, inp, dropout, num_hidden, num_output, noactivation=False,
                 name=None, **kw):
        """Convenience function to combine a MDRNN layer with a voxel-wise fully connected layer.

        :param inp: input data
        :param dropout: dropout rate
        :param num_hidden: number of hidden units, output units of the MDRNN
        :param num_output: number of output units of the voxel-wise fully connected layer
                           (Can be None -> no voxel-wise fully connected layer)
        :param noactivation: Flag to disable activation of voxel-wise fully connected layer
        :param name: Name for this particular MDRNN + vw fully connected layer
        :param kw: Arguments for MDRNN and the vw fully connected layer (can override this class' attributes)
        :return: Output of the voxelwise fully connected layer and MDRNN mix
        """
        dimensions = argget(kw, "dimensions", None)
        if dimensions is None:
            dimensions = [i + 1 for i, v in enumerate(inp.get_shape()[1:-1]) if v > 1]
        mdrnn_kw = {}
        mdrnn_kw.update(self.mdrnn_kw)
        mdrnn_kw.update(self.crnn_kw)
        mdrnn_kw.update(kw)

        add_e_bn = argget(kw, "add_e_bn", self.add_e_bn)
        resmdgru = argget(kw, "resmdgru", self.resmdgru)
        mdrnn_kw["num_hidden"] = num_hidden
        mdrnn_kw["name"] = "mdgru"
        with tf.variable_scope(name):
            mdgruclass = MDRNN(inp, dropout, dimensions, mdrnn_kw)
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
                if self.vwfc_activation in [tf.nn.elu, tf.nn.relu]:
                    numelem = (num_input) / 2
                    uniform = False
                W = tf.get_variable(
                    "W", filtershape, dtype=tf.float32, initializer=get_modified_xavier_method(numelem, uniform))
                b = tf.get_variable("b", [num_output], initializer=tf.constant_initializer(0))

                mdgru = tf.nn.convolution(mdgru, W, padding="SAME")

                if resmdgru:
                    if doreshape:
                        inp = tf.reshape(inp,
                                         [-1, np.prod(inp.get_shape()[1:-1].as_list()), inp.get_shape().as_list()[-1]])
                    resW = tf.get_variable("resW",
                                           [1 for _ in inp.get_shape().as_list()[1:-1]] + [
                                               inp.get_shape().as_list()[-1], num_output],
                                           dtype=tf.float32, initializer=get_modified_xavier_method(num_output, False))
                    mdgru = tf.nn.convolution(inp, resW, padding="SAME") + mdgru
                if add_e_bn:
                    mdgru = batch_norm(mdgru, "bne", mdgruclass.istraining, bias=False, m=mdgruclass.min_mini_batch)
                mdgru = mdgru + b
                if doreshape:
                    mdgru = tf.reshape(mdgru, [-1] + mdgruinnershape + [mdgru.get_shape().as_list()[-1]])
            if noactivation:
                return mdgru
            else:
                return self.vwfc_activation(mdgru)


generate_defaults_info(MDGRUNet)