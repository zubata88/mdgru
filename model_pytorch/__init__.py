__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import numpy as np
from helper import argget
import functools
import copy
import torch as th
from torch.nn import init
import logging


def init_weights(m):
    print(m)
    if hasattr(m, 'initialize'):
        m.initialize()
    else:
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_normal(m.weight.data, gain=0.02)
        elif classname.find('Linear') != -1:
            init.xavier_normal(m.weight.data, gain=0.02)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, 0.02)
            init.constant(m.bias.data, 0.0)
        else:
            print("{} has no method initialize".format(type(m)))

def lazy_property(function):
    """This function computes a property or simply returns it if already computed."""
    attribute = "_" + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return wrapper


# def batch_norm(x, name_scope, training, epsilon=1e-3, decay=0.999, bias=True, m=None):
#     """Computes the batch_norm for x"""
#     with tf.variable_scope(name_scope):
#         size = x.get_shape().as_list()[-1]
#
#         scale = tf.get_variable("scale", [size], initializer=tf.constant_initializer(0.1))
#         if bias:
#             offset = tf.get_variable("offset", [size])
#         else:
#             offset = None
#
#         pop_mean = tf.get_variable("pop_mean", [size], initializer=tf.zeros_initializer(), trainable=False)
#         pop_var = tf.get_variable("pop_var", [size], initializer=tf.constant_initializer(1.0), trainable=False)
#         batch_mean, batch_var = tf.nn.moments(x, [i for i in range(len(x.get_shape()) - 1)])
#
#         # The following simulates a mini-batch for scenarios where we don't have
#         # a large enough mini-batch (THIS ONLY WORKS FOR BATCH SIZES OF 1)
#         if m is not None:
#             batch_mean_list = tf.get_variable("batch_mean_list", [m, size], initializer=tf.zeros_initializer(),
#                                               trainable=False)
#             batch_var_list = tf.get_variable("batch_var_list", [m, size], initializer=tf.constant_initializer(1.0),
#                                              trainable=False)
#             starter = tf.get_variable("batch_list_starter", initializer=tf.constant(0.0), trainable=False,
#                                       dtype=tf.float32)
#
#             def alter_list_at(data, counter, line):
#                 data_unpacked = tf.unstack(data)
#                 data_unpacked.pop(0)
#                 data_unpacked.append(line)
#                 return tf.stack(data_unpacked)
#
#             starter_op = tf.assign(starter, starter + 1)
#             with tf.control_dependencies([starter_op]):
#                 batch_mean_list_op = tf.assign(batch_mean_list, alter_list_at(batch_mean_list, starter, batch_mean))
#                 batch_var_list_op = tf.assign(batch_var_list, alter_list_at(batch_var_list, starter, batch_var))
#             with tf.control_dependencies([batch_mean_list_op, batch_var_list_op]):
#                 batch_mean = tf.reduce_sum(batch_mean_list, 0) / tf.minimum(starter, m)
#                 batch_var = tf.reduce_sum(batch_var_list, 0) / tf.minimum(starter, m - 1)
#
#         def batch_statistics():
#             train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
#             train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
#             with tf.control_dependencies([train_mean_op, train_var_op]):
#                 return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)
#
#         def population_statistics():
#             return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)
#
#         return tf.cond(training, batch_statistics, population_statistics)


class Model(object):
    """Abstract Model class"""

    def __init__(self, data, target, dropout, kw):
        print("model")
        self.origargs = copy.copy(kw)
        # self.model_seed = argget(kw, 'model_seed', 12345678)
        # tf.set_random_seed(self.model_seed)
        super(Model, self).__init__(data, target, dropout, kw)
        # self.training = argget(kw, "training", tf.constant(True))
        # self.global_step = tf.Variable(0, name="global_step", trainable=False)
        # self.use_tensorboard = argget(kw, "use_tensorboard", True)

        if argget(kw, "whiten", False):
            raise Exception('parameter whiten not supported with pytorch version')
            # self.data = batch_norm(data, "bn", self.training, m=32)
        else:
            self.data = data
        pass
        self.dimensions = argget(kw, "dimensions", None)

    def prediction(self):
        """lazy property call to produce a prediction in tensorflow."""
        raise Exception("this should never be called, but implemented by"
                        "the child class")

    def costs(self):
        """lazy property to compute the costs per sample."""
        raise Exception("this should never be called, but implemented by"
                        "the child class")

    @lazy_property
    def cost(self):
        """lazy property to compute the cost per batch"""
        loss = th.mean(self.costs)
        # if self.use_tensorboard:
        #     tf.summary.scalar("loss", loss)
        return loss

    @staticmethod
    def get_model_name_from_ckpt(ckpt):
        """returns root node name of tensorflow graph stored in checkpoint ckpt"""
        raise Exception('this is not supported with pytorch version')
        # try:
        #     r = pywrap_tensorflow.NewCheckpointReader(ckpt)
        #     modelname = r.get_variable_to_shape_map().popitem()[0].split('/')[0]
        # except:
        #     logging.getLogger('runfile').warning('could not load modelname from ckpt-file {}'.format(ckpt))
        #     modelname = None
        # return modelname

    @staticmethod
    def set_allowed_gpu_memory_fraction(gpuboundfraction):
        raise Exception('not supported nor needed with pytorch version')
        # tf.GPUOptions(per_process_gpu_memory_fraction=gpuboundfraction)

class ClassificationModel(Model):
    """Abstract model class. """

    def __init__(self, data, target, dropout, kw):
        print("classificationmodel")
        super(ClassificationModel, self).__init__(data, target, dropout, kw)
        self.target = target
        self.dropout = dropout
        self.learning_rate = argget(kw, "learning_rate", 0.001)
        self.momentum = argget(kw, "momentum", 0.9)
        self.nclasses = argget(kw, "nclasses", 2)




class RegressionModel(Model):
    """Abstract model class for regression tasks."""
    def __init__(self, data, target, dropout, kw):
        super(RegressionModel, self).__init__(data, target, dropout, kw)
        self.target = target
        self.dropout = dropout
        self.learning_rate = argget(kw, "learning_rate", 0.001)
        self.nclasses = argget(kw, "nclasses", 1)
        self.momentum = argget(kw, "momentum", 0.9)


class ReconstructionModel(Model):
    """Abstract model class for reconstruction tasks."""
    def __init__(self, data, dropout, kw):
        super(ReconstructionModel, self).__init__(data, dropout, None, kw)
        self.dropout = dropout
        self.learning_rate = argget(kw, "learning_rate", 0.001)
        self.nclasses = argget(kw, "nclasses", 2)


class GANModel(Model):
    """Abstract model class for GANs."""
    def __init__(self, data, dropout, kw):
        super(GANModel, self).__init__(data, dropout, None, kw)
        self.dropout = dropout
        self.learning_rate = argget(kw, "learning_rate", 0.001)
        self.momentum = argget(kw, "momentum", 0.9)
        self.nclasses = argget(kw, "nclasses", 2)
        self.fakedata = argget(kw, "fakedata", None)
