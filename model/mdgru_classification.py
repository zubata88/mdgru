__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import numpy as np
import tensorflow as tf

from helper import argget, collect_parameters, define_arguments
from model import save_summary_for_nd_images
from model.mdrnn import MDGRUNet
from model.mdrnn.mdgru import MDRNN
from . import ClassificationModel
from . import lazy_property


class MDGRUClassification(ClassificationModel, MDGRUNet):
    """ Provides a full MDGRU default network.

    Using this class,
    Using the parameters fc_channels and mdgru_channels, which have to be lists of the same length, a
    MDGRU network of a number of mdgru and voxel-wise fully connected layers can be generated. Strides can be
    set for each MDGRU layer as a list of lists of strides per dimension. Furthermore, entries in fc_channels may be
    None, when MDGRU layer should be stacked directly after each other.

    :param ignore_label: Implements omitting a certain label, usage of this parameter is discouraged.
    :param fc_channels: Defines the number of channels per voxel-wise fully connected layer
    :param mdgru_channels: Defines the number of channels per MDGRU layer
    :param strides: list of list defining the strides per dimension per MDGRU layer. None means strides of 1
    """

    def __init__(self, data, target, dropout, kw):
        super(MDGRUClassification, self).__init__(data, target, dropout, kw)
        self.ignore_label = argget(kw, "ignore_label", None)
        self.fc_channels = argget(kw, "fc_channels", [25, 45, self.nclasses])
        self.mdgru_channels = argget(kw, "mdgru_channels", [16, 32, 64])
        self.strides = argget(kw, "strides", [None for _ in self.mdgru_channels])

    @lazy_property
    def logits(self):
        """Provides the logits of the prediction"""
        h = self.data
        for it, (mdgruc, fcc, s) in enumerate(zip(self.mdgru_channels, self.fc_channels, self.strides)):
            kw = {}
            if it == len(self.mdgru_channels) - 1:
                kw["bne"] = False
                kw["noactivation"] = True
            if s is not None:
                kw["strides"] = [s for _ in range(len(h.get_shape().as_list()) - 2)] if np.isscalar(s) else s
            h = self.mdgru_bb(h, self.dropout, mdgruc, fcc, name="{}".format(it + 1), istraining=self.training, **kw)
        return h

    @lazy_property
    def prediction(self):
        """Provides prediction in the form of a discrete probability distribution per voxel"""
        pred = tf.nn.softmax(self.logits)
        if self.use_tensorboard:
            save_summary_for_nd_images("data", self.data, collections=["images"])
            save_summary_for_nd_images("prediction", pred, collections=["images"])
        return pred

    @lazy_property
    def costs(self):
        """Cross entropy cost function"""
        if self.ignore_label is not None:
            begin = [0 for _ in len(self.target)]
            size = [-1 for _ in len(self.target)]
            begin[-1] = self.ignore_label
            size[-1] = 1
            ignore = self.target[..., self.ignore_label]
            self.logits *= (1 - tf.expand_dims(ignore, -1))  # this will set all to zero that are going to be ignored
            self.logits[..., self.ignore_label] += ignore * 1e30
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                       labels=self.target)
        if self.ignore_label is not None:
            loss *= tf.size(ignore) / tf.reduce_sum(1 - ignore)
        if self.use_tensorboard:
            save_summary_for_nd_images("target", self.target, collections=["images"])
            tf.summary.scalar("segloss", tf.reduce_mean(loss))
        return loss

    @lazy_property
    def optimize(self):
        """Optimization routine using the Adadelta optimizer"""
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate, rho=self.momentum)
        rets = optimizer.minimize(self.cost, global_step=self.global_step)
        # call prediction to at least initialize everything once:
        self.prediction
        return rets

    @staticmethod
    def collect_parameters():
        args = collect_parameters(MDGRUNet, {})
        args = collect_parameters(MDRNN, args)
        args = collect_parameters(MDRNN._defaults['crnn_class'], args)
        return args
