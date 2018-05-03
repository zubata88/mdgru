__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

from . import ClassificationModel
import tensorflow as tf
from . import lazy_property
from helper import argget, save_summary_for_nd_images
from model.mdrnn import MDGRUNet
import numpy as np


class MDGRUClassification(ClassificationModel, MDGRUNet):
    def __init__(self, data, target, dropout, **kw):
        print("mdgruclassification")
        super(MDGRUClassification, self).__init__(data, target, dropout, **kw)
        self.ignore_label = argget(kw, 'ignore_label', None)
        self.fc_channels = argget(kw, 'fc_channels', [25, 45, self.nclasses])
        self.mdgru_channels = argget(kw, 'mdgru_channels', [16, 32, 64])
        self.strides = argget(kw, 'strides', [None for _ in self.mdgru_channels])

    @lazy_property
    def logits(self):
        h = self.data
        for it, (mdgruc, fcc, s) in enumerate(zip(self.mdgru_channels, self.fc_channels, self.strides)):
            kw = {}
            if it == len(self.mdgru_channels) - 1:
                kw['bne'] = False
                kw['noactivation'] = True
            if s is not None:
                kw['strides'] = [s for _ in range(len(h.get_shape().as_list()) - 2)] if np.isscalar(s) else s
            h = self.mdgru_bb(h, self.dropout, mdgruc, fcc, name="{}".format(it + 1), istraining=self.training, **kw)
        return h

    @lazy_property
    def prediction(self):
        pred = tf.nn.softmax(self.logits)
        if self.use_tensorboard:
            save_summary_for_nd_images('data', self.data, collections=['images'])
            save_summary_for_nd_images('prediction', pred, collections=['images'])
        return pred


    @lazy_property
    def costs(self):
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
            save_summary_for_nd_images('target', self.target, collections=['images'])
            tf.summary.scalar('segloss', tf.reduce_mean(loss))
        return loss

    @lazy_property
    def optimize(self):
        '''Optimize the model.'''

        #go for the optimization
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate, rho=self.momentum)
        rets = optimizer.minimize(self.cost, global_step=self.global_step)
        #call prediction to at least initialize everything:
        self.prediction
        return rets

