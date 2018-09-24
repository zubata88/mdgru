__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import numpy as np
from mdgru.helper import argget, compile_arguments
import functools
import copy
import torch as th
from torch.nn import init
import logging


def init_weights(m):
    if hasattr(m, 'initialize_weights'):
        m.initialize_weights()
    else:
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and hasattr(m, 'weight'):
            init.xavier_normal_(m.weight.data)
            if hasattr(m, 'bias'):
                init.constant_(m.bias.data, 0.0)
        else:
            pass #if not in above list, parameters wont be initialized using the initialization rule.


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


class Model(th.nn.Module):
    """Abstract Model class"""

    _defaults = {
        "model_seed": {'value': 12345678, 'help': "Override default model initialization random seed"},
    }

    def __init__(self, data, dropout, kw):
        super(Model, self).__init__()
        model_kw, kw = compile_arguments(Model, kw, transitive=False)
        for k, v in model_kw.items():
            setattr(self, k, v)
        th.cuda.manual_seed_all(self.model_seed)
        th.manual_seed(self.model_seed)
        self.origargs = copy.copy(kw)

        if argget(kw, "whiten", False):
            print('parameter whiten not supported with pytorch version')
            # self.data = batch_norm(data, "bn", self.training, m=32)
        else:
            self.data = data
        pass
        self.dimensions = argget(kw, "dimensions", None)

    @staticmethod
    def get_model_name_from_ckpt(_):
        """returns root node name of tensorflow graph stored in checkpoint ckpt"""
        return 'default'

    @staticmethod
    def set_allowed_gpu_memory_fraction(gpuboundfraction):
        raise Exception('not supported nor needed with pytorch version')
        # tf.GPUOptions(per_process_gpu_memory_fraction=gpuboundfraction)


class ClassificationModel(Model):
    """Abstract model class. """
    def __init__(self, data, dropout, kw):
        super(ClassificationModel, self).__init__(data, dropout, kw)
        self.dropout = dropout
        self.learning_rate = argget(kw, "learning_rate", 1)
        self.momentum = argget(kw, "momentum", 0.9)
        self.nclasses = argget(kw, "nclasses", 2)


class RegressionModel(Model):
    """Abstract model class for regression tasks."""
    def __init__(self, data, dropout, kw):
        super(RegressionModel, self).__init__(data, dropout, kw)
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
