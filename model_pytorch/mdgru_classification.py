__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import numpy as np
import tensorflow as tf

from helper import argget, save_summary_for_nd_images
from model_pytorch.mdrnn import MDGRUBlock
from . import ClassificationModel
from . import lazy_property
import torch as th

class MDGRUClassification(ClassificationModel):
    """ Provides a full MDGRU default network.

    Using this class,
    Using the parameters fc_channels and mdgru_channels, which have to be lists of the same length, a
    MDGRU network of a number of mdgru and voxel-wise fully connected layers can be generated. Strides can be
    set for each MDGRU layer as a list of lists of strides per dimension. Furthermore, entries in fc_channels may be
    None, when MDGRU layer should be stacked directly after each other.

    :param fc_channels: Defines the number of channels per voxel-wise fully connected layer
    :param mdgru_channels: Defines the number of channels per MDGRU layer
    :param strides: list of list defining the strides per dimension per MDGRU layer. None means strides of 1
    :param data_shape: subvolume size
    """

    def __init__(self, data_shape, nclasses, dropout, kw):
        super(MDGRUClassification, self).__init__(data_shape, nclasses, dropout, kw)
        self.fc_channels = argget(kw, "fc_channels", [25, 45, self.nclasses])
        self.mdgru_channels = argget(kw, "mdgru_channels", [16, 32, 64])
        self.strides = argget(kw, "strides", [None for _ in self.mdgru_channels])
        self.data_shape = data_shape
        #create logits:
        logits = []
        num_spatial_dims = len(data_shape[2:])
        last_output_channel_size = data_shape[1]
        for it, (mdgru, fcc, s) in enumerate(zip(self.mdgru_channels, self.fc_channels, self.strides)):
            mdgru_kw = {}
            mdgru_kw.update(kw)
            if it == len(self.mdgru_channels) - 1:
                mdgru_kw["noactivation"] = True
            if s is not None:
                mdgru_kw["strides"] = [s for _ in range(num_spatial_dims)] if np.isscalar(s) else s
            logits += [MDGRUBlock(num_spatial_dims, self.dropout, last_output_channel_size, mdgru, fcc, name="{}".format(it + 1), **kw)]
            last_output_channel_size = fcc if fcc is not None else mdgru
        self.logits = th.nn.Sequential(*logits)

    def prediction(self):
        """Provides prediction in the form of a discrete probability distribution per voxel"""
        pred = th.NN.softmax(self.logits)
        return pred

    def costs(self):
        """Cross entropy cost function"""
        loss = th.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                       labels=self.target)
        return loss

    def optimize(self):
        """Optimization routine using the Adadelta optimizer"""
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate, rho=self.momentum)
        rets = optimizer.minimize(self.cost, global_step=self.global_step)
        # call prediction to at least initialize everything once:
        return rets
