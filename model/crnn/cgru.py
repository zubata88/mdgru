import logging
from copy import deepcopy

import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from helper import argget
from model import batch_norm
from model.crnn import CRNNCell


class CGRUCell(CRNNCell):
    def __init__(self, *w, **kw):
        self.bnx = argget(kw, "add_x_bn", False)
        self.bnh = argget(kw, "add_h_bn", False)
        self.bna = argget(kw, "add_a_bn", False)
        self.m = argget(kw, "m", None)
        self.istraining = argget(kw, 'istraining', tf.constant(True))
        self.resgrux = argget(kw, "resgrux", False)
        self.resgruh = argget(kw, "resgruh", False)
        self.put_r_back = argget(kw, "put_r_back", False)
        self.regularize_state = argget(kw, 'use_dropconnect_on_state', False)

        super(CGRUCell, self).__init__(*w, **kw)

        filterhshapecandidate = self.filter_size_h + [self._num_units] * 2
        filterhshapegates = deepcopy(filterhshapecandidate)
        filterhshapegates[-1] *= 2

        filterxshapecandidate = self.filter_size_x + [self.myshapes[0][-1], self._num_units]
        filterxshapegates = deepcopy(filterxshapecandidate)
        filterxshapegates[-1] *= 2

        self.dropconnecthmatrixgates = self._get_dropconnect(filterhshapegates, self.dropconnecth,
                                                             "mydropconnecthgates")
        self.dropconnecthmatrixcandidate = self._get_dropconnect(filterhshapecandidate, self.dropconnecth,
                                                                 "mydropconnecthcandidate")

        self.dropconnectxmatrixgates = self._get_dropconnect(filterxshapegates, self.dropconnectx,
                                                             "mydropconnectxgates")
        self.dropconnectxmatrixcandidate = self._get_dropconnect(filterxshapecandidate, self.dropconnectx,
                                                                 "mydropconnectxcandidate")

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with nunits cells.
        
        """
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                zrx, zrh, zrb = self._convlinear([inputs, state], self._num_units * 2, True, 1.0,
                                                 dropconnectx=self.dropconnectx, dropconnecth=self.dropconnecth,
                                                 dropconnectxmatrix=self.dropconnectxmatrixgates,
                                                 dropconnecthmatrix=self.dropconnecthmatrixgates,
                                                 strides=self.strides, orthogonal_init=False)
            if self.bnx:
                zrx = batch_norm(zrx, "bnx", self.istraining, bias=None, m=self.m)
            if self.bnh:
                zrh = batch_norm(zrh, "bnh", self.istraining, bias=None, m=self.m)
            with vs.variable_scope('resgru'):
                if self.resgrux:
                    ntiles = self._num_units // inputs.get_shape().as_list()[-1]
                    if ntiles * inputs.get_shape().as_list()[-1] != self._num_units:
                        logging.getLogger('model').error(
                            'cant do resgrux here, since {}*{} is smaller than the actual number of outputs. (needs to be multiple){}'
                                .format(ntiles, inputs.get_shape().as_list()[-1], self._num_units))
                    else:
                        zrx = tf.tile(inputs, [1, ntiles * 2]) + zrx
                if self.resgruh:
                    zrh = tf.tile(state, [1, 2]) + zrh
            z, r = tf.split(zrx + zrh + zrb, 2, axis=len(inputs.get_shape()) - 1)
            z, r = self.gate(z), self.gate(r)
            with vs.variable_scope("Candidate"):
                if self.put_r_back:
                    state *= r
                usedx = self.dropconnectx if self.regularize_state else None
                usedh = self.dropconnecth if self.regularize_state else None
                htx, hth, htb = self._convlinear([inputs, state],
                                                 self._num_units, True,
                                                 dropconnectxmatrix=self.dropconnectxmatrixcandidate,
                                                 dropconnecthmatrix=self.dropconnecthmatrixcandidate,
                                                 dropconnectx=usedx, dropconnecth=usedh,
                                                 strides=self.strides)
                if self.put_r_back:
                    htwb = htx + hth
                else:
                    htwb = htx + r * hth
            if self.bna:
                htwb = batch_norm(htwb, "bna", self.istraining, bias=None, m=self.m)

            new_h = z * state + (1 - z) * self._activation(htwb + htb)
        return new_h, new_h
