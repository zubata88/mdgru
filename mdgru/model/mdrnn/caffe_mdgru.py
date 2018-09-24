__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

# Warning: this file is obsolete and not maintained anymore.

from helper import argget
import copy
import tensorflow as tf
from tensorflow_extra_ops import CaffeBiCGRU3D

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
        self.istraining = argget(kw, "istraining", tf.constant(True))
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
        self.use_bernoulli = argget(kw, "use_bernoulli_dropconnect", False)
        self.mask_padding = argget(kw, "maskpadding", None)
        self.m = argget(kw, "min_mini_batch", None)
        self.favor_speed_over_memory = argget(kw, "favor_speed_over_memory", False)
        self.filter_sizes = argget(kw, "filter_sizes", [7, 7, 7])
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