__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import numpy as np
import tensorflow as tf

from mdgru.helper import argget, collect_parameters, define_arguments, compile_arguments
from mdgru.model import save_summary_for_nd_images
from mdgru.model.mdrnn import MDGRUNet
from mdgru.model.mdrnn.mdgru import MDRNN
from mdgru.model import ClassificationModel
from mdgru.model import lazy_property


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
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
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

    @staticmethod
    def compile_arguments(kw, keep_entries=True):
        # block_kw, kw = compile_arguments(MDGRUBlock, kw, transitive=True)
        mdrnn_kw, kw = compile_arguments(MDRNN, kw, transitive=True, keep_entries=keep_entries)
        crnn_kw, kw = compile_arguments(MDRNN._defaults['crnn_class'], kw, transitive=True, keep_entries=keep_entries)
        new_kw = {}
        new_kw.update(crnn_kw)
        new_kw.update(mdrnn_kw)
        # new_kw.update(block_kw)
        return new_kw, kw

class MDGRUClassificationWithDiceLoss(MDGRUClassification):
    def __init__(self, data, target, dropout, kw):
        super(MDGRUClassificationWithDiceLoss, self).__init__(data, target, dropout, kw)
        self.dice_loss_label = argget(kw, "dice_loss_label", [])
        self.dice_loss_weight = argget(kw, "dice_loss_weight", [])
        self.dice_autoweighted = argget(kw, "dice_autoweighted", False)

        if len(self.dice_loss_label) != len(self.dice_loss_weight) and not self.dice_autoweighted:
            raise Exception("dice_loss_label and dice_loss_weight need to be of the same length")

    @lazy_property
    def costs(self):
        # get standard entropy loss
        crossEntropyLoss = super(MDGRUClassificationWithDiceLoss, self).costs

        shape = self.prediction.get_shape()
        ndim = len(shape)
        area = np.prod(shape[1:ndim - 1])
        batch_size = tf.shape(self.prediction)[0]
        eps = 1e-8

        # calc soft dice loss, loop over all declared dice loss labels
        diceLoss = 0
        if self.dice_autoweighted:
            batchDiceLoss = tf.zeros([batch_size])
            batchtotalWeight = tf.zeros([batch_size])
            for l in self.dice_loss_label:
                intersection =      tf.reduce_sum(self.prediction[..., l] * self.target[..., l], [i for i in range(1, ndim - 1)])
                sum_prediction =    tf.reduce_sum(self.prediction[..., l], [i for i in range(1, ndim - 1)])
                sum_target =        tf.reduce_sum(self.target[..., l], [i for i in range(1, ndim - 1)])
                w_all_batches = 1 / (tf.square(sum_target) + 1) # to prevent infty if label is not in the sample
                batchtotalWeight += w_all_batches
                batchDiceLoss += w_all_batches * (2 * intersection + eps) / (sum_prediction + sum_target + eps)
            diceLoss = - sum(self.dice_loss_weight) * tf.reduce_mean(batchDiceLoss / batchtotalWeight)
        elif sum(self.dice_loss_weight) > 0:
            for w, l in zip(self.dice_loss_weight, self.dice_loss_label):
                intersection =      tf.reduce_sum(self.prediction[..., l] * self.target[..., l], [i for i in range(1, ndim - 1)])
                sum_prediction =    tf.reduce_sum(self.target[..., l], [i for i in range(1, ndim - 1)])
                sum_target =        tf.reduce_sum(self.prediction[..., l], [i for i in range(1, ndim - 1)])
                diceLoss -= w * tf.reduce_mean((2 * intersection + eps) / (sum_prediction + sum_target + eps))

        return diceLoss + (1 - sum(self.dice_loss_weight)) * crossEntropyLoss


class MDGRUClassificationWithGeneralizedDiceLoss(MDGRUClassification):
    def __init__(self, data, target, dropout, kw):
        super(MDGRUClassificationWithGeneralizedDiceLoss, self).__init__(data, target, dropout, kw)
        self.dice_loss_label = argget(kw, "dice_loss_label", [])
        self.dice_loss_weight = argget(kw, "dice_loss_weight", [])
        self.dice_autoweighted = argget(kw, "dice_autoweighted", False)

        if len(self.dice_loss_label) != len(self.dice_loss_weight) and not self.dice_autoweighted:
            raise Exception("dice_loss_label and dice_loss_weight need to be of the same length")

    @lazy_property
    def costs(self):
        # get standard entropy loss
        crossEntropyLoss = super(MDGRUClassificationWithGeneralizedDiceLoss, self).costs

        shape = self.prediction.get_shape()
        ndim = len(shape)
        batch_size = tf.shape(self.prediction)[0]
        eps = 1e-8

        # calc soft dice loss, loop over all declared dice loss labels
        diceLoss = 0
        total_intersection = tf.zeros([batch_size])
        total_sum = tf.zeros([batch_size])
        if self.dice_autoweighted:
            for l in self.dice_loss_label:
                sum_prediction =        tf.reduce_sum(self.prediction[..., l], [i for i in range(1, ndim - 1)])
                sum_target =            tf.reduce_sum(self.target[..., l], [i for i in range(1, ndim - 1)])
                w_all_batches = 1 / (tf.square(sum_target) + 1) # to prevent infty if label is not in the sample
                total_sum += w_all_batches * (sum_prediction + sum_target)
                total_intersection += w_all_batches * tf.reduce_sum(self.prediction[..., l] * self.target[..., l], [i for i in range(1, ndim - 1)])
            diceLoss = - sum(self.dice_loss_weight) * tf.reduce_mean((2 * total_intersection + eps) / (total_sum + eps))
        elif sum(self.dice_loss_weight) > 0:
            for w, l in zip(self.dice_loss_weight, self.dice_loss_label):
                total_intersection +=     w * tf.reduce_sum(self.prediction[..., l] * self.target[..., l], [i for i in range(1, ndim - 1)])
                total_sum +=    w * (tf.reduce_sum(self.target[..., l], [i for i in range(1, ndim - 1)]) + tf.reduce_sum(self.prediction[..., l], [i for i in range(1, ndim - 1)]))
            diceLoss = -tf.reduce_mean((2 * total_intersection + eps) / (total_sum + eps))

        return diceLoss + (1 - sum(self.dice_loss_weight)) * crossEntropyLoss
