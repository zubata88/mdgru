__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import logging
from copy import deepcopy
from helper import compile_arguments
# from model_pytorch import batch_norm
from model_pytorch.crnn import CRNNCell
import torch as th
from torch.nn.parameter import Parameter
from torch import functional as F

class CGRUCell(CRNNCell):
    """Convolutional gated recurrent unit.

    This class processes n-d data along the last dimension using a gated recurrent unit, which uses n-1 d convolutions
    on its path along that last dimension to gather context from input and last state to produce the new state.
    Property defaults contains defaults for all properties of a CGRUCell that are the same for one MDGRU.
    :param add_x_bn: Enables batch normalization on inputs for gates
    :param add_h_bn: Enables batch normalization on last state for gates
    :param add_a_bn: Enables batch normalization for the candidate (both input and last state)
    :param resgrux: Enables residual learning on weighted input
    :param resgruh: Enables residual learning on weighted previous output / state
    :param use_dropconnect_on_state: Should dropconnect be used also for the candidate computation?
    :param put_r_back: Use reset gate r's position of original gru formulation, which complicates computation.
    :param min_mini_batch: Emulation
    :param gate: Defines activation function to be used for gates
    """
    _defaults = {
        "add_x_bn": False,
        "add_h_bn": False,
        "add_a_bn": False,
        "resgrux": False,
        "resgruh": False,
        "put_r_back": False,
        "use_dropconnect_on_state": False,
        "min_mini_batch": False,
        # "istraining": th.constant(True),
        "gate": th.sigmoid,
        "learnable_state": False
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
        self.filter_shape_x_candidate = [self._num_inputs, self._num_units] + self.filter_size_x
        self.filter_shape_x_gates = deepcopy(self.filter_shape_x_candidate)
        self.filter_shape_x_gates[0] *= 2

        self.filter_x_gates = Parameter(th.Tensor(*self.filter_shape_x_gates))
        self.filter_x_candidate = Parameter(th.Tensor(*self.filter_shape_x_candidate))
        self.filter_h_gates = Parameter(th.Tensor(*self.filter_shape_h_gates))
        self.filter_h_candidate = Parameter(th.Tensor(*self.filter_shape_h_candidate))

        self.bias_x_gates = Parameter(th.Tensor(self.filter_shape_x_gates[0]))
        self.bias_x_candidate = Parameter(th.Tensor(self.filter_shape_x_candidate[0]))
        self.bias_h_gates = Parameter(th.Tensor(self.filter_shape_h_gates[0]))
        self.bias_h_candidate = Parameter(th.Tensor(self.filter_shape_h_candidate[0]))

        if self.dropconnecth is not None:
            self.dropconnect_h_gates = th.zeros_like(self.filter_h_gates)
        if self.dropconnectx is not None:
            self.dropconnect_x_gates = th.zeros_like(self.filter_x_gates)
        if self.use_dropconnect_on_state:
            if self.dropconnecth is not None:
                self.dropconnect_h_candidate = th.zeros_like(self.filter_h_candidate)
            if self.dropconnectx is not None:
                self.dropconnect_x_candidate = th.zeros_like(self.filter_x_candidate)

        if self.learnable_state:
            raise Exception("this is not yet supported")
        # else:
        #     self.initial_state = th.Tensor(*self.)
        self.reset_parameters()

    def forward(self, inputs):
        weights_h_gates = self.filter_h_gates
        weights_h_candidate = self.filter_h_candidate
        weights_x_gates = self.filter_x_gates
        weights_x_candidate = self.filter_x_candidate

        if self.training: #TODO: add option to apply dropout during inference!
            if self.dropconnecth is not None:
                self._get_dropconnect(self.dropconnect_h_gates, self.dropconnecth)
                weights_h_gates = weights_h_gates * self.dropconnect_h_gates
            if self.dropconnectx is not None:
                self._get_dropconnect(self.dropconnect_x_gates, self.dropconnectx)
                weights_x_gates = weights_x_gates * self.dropconnect_x_gates
            if self.use_dropconnect_on_state:
                if self.dropconnecth is not None:
                    self._get_dropconnect(self.dropconnect_h_candidate, self.dropconnecth)
                    weights_h_candidate = weights_h_candidate * self.dropconnect_h_candidate
                if self.dropconnectx is not None:
                    self._get_dropconnect(self.dropconnect_x_candidate, self.dropconnectx)
                    weights_x_candidate = weights_x_candidate * self.dropconnect_x_candidate
            state = None
        # stride = [1] * self.num_spatial_dims
        padding_x = [f//2 for f in self.filter_size_x]
        padding_h = [f//2 for f in self.filter_size_h]
        # dilation = [1] * self.num_spatial_dims
        # groups = [1] * self.num_spatial_dims
        states = []
        for i, inp in enumerate(inputs):
            zrxb = self.convop(inp, weights_x_gates, self.bias_x_gates, padding=padding_x)
            htxb = self.convop(inp, weights_x_candidate, self.bias_x_candidate, padding=padding_x)
            if i > 0:
                zrh = self.convop(state, weights_h_gates, padding=padding_h)
                hth = self.convop(state, weights_h_candidate, padding=padding_h)
                prev_state = states[i - 1]
            else:
                zrh, hth, prev_state = 0, 0, 0
            z, r = th.split(self.gate(zrxb + zrh), 2, 1)
            ht = self.crnn_activation(htxb + r * hth)
            states.append(z * prev_state + (1-z) * ht)
        return states

    #
    #
    # def __call__(self, inputs, state, scope=None):
    #     """Perform one timestep of this cGRU.
    #
    #     :param inputs: Input at timestep t.
    #     :param state: State at timestep t-1.
    #     :param scope: Optional named scope.
    #     :return: State at timestep t.
    #     """
    #     with vs.variable_scope(scope or type(self).__name__):
    #         with vs.variable_scope("Gates"):  # Reset gate and update gate.
    #             # We start with bias of 1.0 to not reset and not update.
    #             zrx, zrh, zrb = self._convlinear([inputs, state], self._num_units * 2, True, 1.0,
    #                                              dropconnectx=self.dropconnectx, dropconnecth=self.dropconnecth,
    #                                              dropconnectxmatrix=self.dc_x_factor_gates,
    #                                              dropconnecthmatrix=self.dc_h_factor_gates,
    #                                              strides=self.strides, orthogonal_init=False)
    #         # Perform batch norm on x and/or h of the gates if needed.
    #         if self.add_x_bn:
    #             zrx = batch_norm(zrx, "bnx", self.istraining, bias=None, m=self.min_mini_batch)
    #         if self.add_h_bn:
    #             zrh = batch_norm(zrh, "bnh", self.istraining, bias=None, m=self.min_mini_batch)
    #         # Add skip connections for input x and/or state h of the gates if needed.
    #         with vs.variable_scope("resgru"):
    #             if self.resgrux:
    #                 ntiles = self._num_units // inputs.get_shape().as_list()[-1]
    #                 if ntiles * inputs.get_shape().as_list()[-1] != self._num_units:
    #                     logging.getLogger("model").error(
    #                         "{}*{} is smaller than the actual number of outputs. (needs to be multiple){}"
    #                             .format(ntiles, inputs.get_shape().as_list()[-1], self._num_units))
    #                 else:
    #                     zrx = tf.tile(inputs, [1, ntiles * 2]) + zrx
    #             if self.resgruh:
    #                 zrh = tf.tile(state, [1, 2]) + zrh
    #         # Separate and activate update gate z and reset gate r
    #         z, r = tf.split(zrx + zrh + zrb, 2, axis=len(inputs.get_shape()) - 1)
    #         z, r = self.gate(z), self.gate(r)
    #         # Compute candidate \tilde{h}
    #         with vs.variable_scope("Candidate"):  # Proposal or Candidate.
    #             if self.put_r_back:
    #                 state *= r
    #             usedx = self.dropconnectx if self.use_dropconnect_on_state else None
    #             usedh = self.dropconnecth if self.use_dropconnect_on_state else None
    #             htx, hth, htb = self._convlinear([inputs, state],
    #                                              self._num_units, True,
    #                                              dropconnectxmatrix=self.dc_x_factor_candidate,
    #                                              dropconnecthmatrix=self.dc_h_factor_candidate,
    #                                              dropconnectx=usedx, dropconnecth=usedh,
    #                                              strides=self.strides)
    #             if self.put_r_back:
    #                 htwb = htx + hth
    #             else:
    #                 htwb = htx + r * hth
    #         # Perform batch norm on candidate if needed.
    #         if self.add_a_bn:
    #             htwb = batch_norm(htwb, "bna", self.istraining, bias=None, m=self.min_mini_batch)
    #         # Update state/output.
    #         new_h = z * state + (1 - z) * self.crnn_activation(htwb + htb)
    #     return new_h, new_h
