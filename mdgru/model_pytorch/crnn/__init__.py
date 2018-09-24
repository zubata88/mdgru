__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import logging
from copy import deepcopy

import numpy as np
import torch as th

from helper import argget, compile_arguments


class CRNNCell(th.nn.Module):
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
        "crnn_activation": th.tanh,
    }

    def __init__(self, num_input, num_units, kw):
        super(CRNNCell, self).__init__()
        crnn_kw, kw = compile_arguments(CRNNCell, kw, transitive=False)
        for k, v in crnn_kw.items():
            setattr(self, k, v)
        self._num_units = num_units
        self._num_inputs = num_input
        self.filter_size_x = argget(kw, "filter_size_x", [7, 7])
        self.filter_size_h = argget(kw, "filter_size_h", [7, 7])
        self.strides = argget(kw, "strides", None)

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def _get_dropconnect(self, t, keep_rate_training, keep_rate_testing=1):
        """Creates factors to be applied to filters to achieve either Bernoulli or Gaussian dropconnect."""
        if self.training:
            keep_rate = keep_rate_training
        else:
            keep_rate = keep_rate_testing
        if keep_rate is None:
            raise Exception('keeprate cannot be none if this is called')
        if self.use_bernoulli:
            dc = t.random_() < keep_rate
            t.fill_(1).mul_(dc).mul_(1/keep_rate)
        else:
            t.normal_(1, np.sqrt((1 - keep_rate) / keep_rate))


