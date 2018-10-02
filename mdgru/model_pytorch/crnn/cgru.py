__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import logging
from copy import deepcopy
from mdgru.helper import compile_arguments, generate_defaults_info
from mdgru.model_pytorch.crnn import CRNNCell
import torch as th
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F

class CGRUCell(CRNNCell):
    """Convolutional gated recurrent unit.

    This class processes n-d data along the last dimension using a gated recurrent unit, which uses n-1 d convolutions
    on its path along that last dimension to gather context from input and last state to produce the new state.
    Property defaults contains defaults for all properties of a CGRUCell that are the same for one MDGRU.
    """
    _defaults = {
        "put_r_back": {'value': False, 'help': 'Place reset gate at its original location, as in the original GRU'},
        "use_dropconnect_on_state": {'value': False, 'help': 'Apply dropconnect regularization also to the proposal, not only the gates'},
        "gate": {'value': th.sigmoid, 'help': 'Gate activation function to use'},
        "start_state": None
    }

    def __init__(self, num_input, num_units, kw):
        super(CGRUCell, self).__init__(num_input, num_units, kw)
        cgru_kw, kw = compile_arguments(CGRUCell, kw, transitive=False)
        for k, v in cgru_kw.items():
            setattr(self, k, v)
        self.num_spatial_dims = len(self.filter_size_x)
        if self.num_spatial_dims == 2:
            self.convop = F.conv2d
        elif self.num_spatial_dims == 1:
            self.convop = F.conv1d
        else:
            raise Exception("convolutions of size {} are not implemented in pytorch cgru")
        self.filter_shape_h_candidate = [self._num_units] * 2 + self.filter_size_h
        self.filter_shape_h_gates = deepcopy(self.filter_shape_h_candidate)
        self.filter_shape_h_gates[0] *= 2
        self.filter_shape_x_candidate = [self._num_units, self._num_inputs] + self.filter_size_x
        self.filter_shape_x_gates = deepcopy(self.filter_shape_x_candidate)
        self.filter_shape_x_gates[0] *= 2

        self.filter_x_gates = Parameter(th.Tensor(*self.filter_shape_x_gates))
        self.filter_x_candidate = Parameter(th.Tensor(*self.filter_shape_x_candidate))
        self.filter_h_gates = Parameter(th.Tensor(*self.filter_shape_h_gates))
        self.filter_h_candidate = Parameter(th.Tensor(*self.filter_shape_h_candidate))

        self.bias_x_gates = Parameter(th.Tensor(self.filter_shape_x_gates[0]))
        self.bias_x_candidate = Parameter(th.Tensor(self.filter_shape_x_candidate[0]))

        if self.dropconnecth is not None:
            self.register_buffer("dropconnect_h_gates", th.zeros_like(self.filter_h_gates))
        if self.dropconnectx is not None:
            self.register_buffer("dropconnect_x_gates", th.zeros_like(self.filter_x_gates))
        if self.use_dropconnect_on_state:
            if self.dropconnecth is not None:
                self.register_buffer("dropconnect_h_candidate", th.zeros_like(self.filter_h_candidate))
            if self.dropconnectx is not None:
                self.register_buffer("dropconnect_x_candidate", th.zeros_like(self.filter_x_candidate))

    def initialize_weights(self):
        self.bias_x_gates.data.fill_(1)
        self.bias_x_candidate.data.fill_(0)
        init.xavier_normal_(self.filter_x_gates.data)
        init.xavier_normal_(self.filter_x_candidate.data)
        init.xavier_normal_(self.filter_h_gates.data)
        init.xavier_normal_(self.filter_h_candidate.data)

    def forward(self, inputs):
        weights_h_gates = self.filter_h_gates
        weights_h_candidate = self.filter_h_candidate
        weights_x_gates = self.filter_x_gates
        weights_x_candidate = self.filter_x_candidate

        if self.training: #TODO: add option to apply dropout during inference!
            if self.dropconnecth is not None:
                self._get_dropconnect(self.dropconnect_h_gates, self.dropconnecth)
                weights_h_gates = self.filter_h_gates * self.dropconnect_h_gates
            if self.dropconnectx is not None:
                self._get_dropconnect(self.dropconnect_x_gates, self.dropconnectx)
                weights_x_gates = self.filter_x_gates * self.dropconnect_x_gates
            if self.use_dropconnect_on_state:
                if self.dropconnecth is not None:
                    self._get_dropconnect(self.dropconnect_h_candidate, self.dropconnecth)
                    weights_h_candidate = self.filter_h_candidate * self.dropconnect_h_candidate
                if self.dropconnectx is not None:
                    self._get_dropconnect(self.dropconnect_x_candidate, self.dropconnectx)
                    weights_x_candidate = self.filter_x_candidate * self.dropconnect_x_candidate
        # stride = [1] * self.num_spatial_dims
        padding_x = [f//2 for f in self.filter_size_x]
        padding_h = [f//2 for f in self.filter_size_h]
        # dilation = [1] * self.num_spatial_dims
        # groups = [1] * self.num_spatial_dims
        states = []
        prev_state = self.start_state
        for i, inp in enumerate(inputs):
            zrxb = self.convop(inp, weights_x_gates, self.bias_x_gates, padding=padding_x)
            htxb = self.convop(inp, weights_x_candidate, self.bias_x_candidate, padding=padding_x)
            if prev_state is not None:
                zrh = self.convop(prev_state, weights_h_gates, padding=padding_h)
                hth = self.convop(prev_state, weights_h_candidate, padding=padding_h)
            else:
                zrh, hth, prev_state = 0, 0, 0
            zr = self.gate(zrxb + zrh)
            z, r = th.split(zr, self._num_units, 1)
            ht = self.crnn_activation(htxb + r * hth)
            states.append(z * prev_state + (1 - z) * ht)
            # For next iteration:
            prev_state = states[i]
        return states


generate_defaults_info(CGRUCell)