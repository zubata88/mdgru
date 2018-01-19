__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import tensorflow as tf
import numpy as np
from helper import argget
import functools
import copy


def lazy_property(function):
    ''' This function acts at its first call as a function, each further call to
    the function just returns the computed return value as property.
    '''
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


def layer_norm(x, s=None, b=None, use_bias=False,eps = 1e-8):
    if s is None:
        s = tf.get_variable('lnscale',initializer=tf.constant_initializer(1.0),shape=[1])
    m, v = tf.nn.moments(x, [i for i in range(1,len(x.get_shape()))], keep_dims=True)
    nx = (x - m) / tf.sqrt(v + eps)
    nx *= s
    if use_bias:
        if b is None:
            b = tf.get_variable('lnbias',initializer=tf.constant_initializer(0.0),shape=[1])
        nx+=b
    return nx


def batch_norm(x, name_scope, training, epsilon=1e-3, decay=0.999,bias=True,m=None):
    '''Assume 2d [batch, values] tensor'''

    with tf.variable_scope(name_scope):
        size = x.get_shape().as_list()[-1]

        scale = tf.get_variable('scale', [size], initializer=tf.constant_initializer(0.1))
        if bias:
            offset = tf.get_variable('offset', [size])
        else:
            offset = None

        pop_mean = tf.get_variable('pop_mean', [size], initializer=tf.zeros_initializer(), trainable=False)
        pop_var = tf.get_variable('pop_var', [size], initializer=tf.constant_initializer(1.0), trainable=False)
        batch_mean, batch_var = tf.nn.moments(x, [i for i in range(len(x.get_shape())-1)])

        # The following simulates a mini-batch for scenarios where we dont have 
        # a large enough mini-batch (THIS ONLY WORKS FOR BATCH SIZES OF 1)
        if m is not None:
            batch_mean_list = tf.get_variable('batch_mean_list', [m, size], initializer=tf.zeros_initializer(), trainable=False)
            batch_var_list = tf.get_variable('batch_var_list', [m, size], initializer=tf.constant_initializer(1.0), trainable=False)
            starter = tf.get_variable('batch_list_starter', initializer=tf.constant(0.0), trainable=False, dtype=tf.float32)

            def alter_list_at(data,counter,line):
                data_unpacked = tf.unstack(data)
                data_unpacked.pop(0)
                data_unpacked.append(line)
                return tf.stack(data_unpacked)
            starter_op = tf.assign(starter,starter+1)
            with tf.control_dependencies([starter_op]):
                batch_mean_list_op = tf.assign(batch_mean_list, alter_list_at(batch_mean_list, starter, batch_mean))
                batch_var_list_op = tf.assign(batch_var_list, alter_list_at(batch_var_list, starter, batch_var))
            with tf.control_dependencies([batch_mean_list_op, batch_var_list_op]):
                batch_mean = tf.reduce_sum(batch_mean_list, 0)/tf.minimum(starter, m)
                batch_var = tf.reduce_sum(batch_var_list, 0)/tf.minimum(starter, m-1)

        def batch_statistics():
            train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)

        def population_statistics():
            return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

        return tf.cond(training, batch_statistics, population_statistics)


class Model(object):
    '''Abstract model class. '''

    def __init__(self, data, target=None, dropout=None, **kw):
        print("model")
        self.origargs = copy.copy(kw)
        self.l2 = argget(kw, 'show_l2_loss', True)

        tf.set_random_seed(12345678)
        super(Model,self).__init__(data,target,dropout,**kw)
        self.training = argget(kw,"training",tf.constant(True))
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.use_tensorboard = argget(kw, "use_tensorboard", True, keep=True)

        if argget(kw,'whiten',False):
            self.data = batch_norm(data, "bn", self.training, m=32)
        else:
            self.data = data
        pass
        self.dimensions = argget(kw, "dimensions", None)
#     def batch_norm(self,inp):
#         mean = tf.reduce_mean(inp)
#         self.running_num = self.running_num*0.999+1
#         self.running_mean = self.running_mean*0.999+mean
#         inp -= self.running_mean/self.running_num
#         var = tf.reduce_mean(inp*inp)
#         self.running_var = self.running_var*0.999+var
#         inp /= tf.sqrt(self.running_var/self.running_num+1e-20)
#         return inp
    def prediction(self):
        '''lazy property call to produce a prediction in tensorflow.'''
        raise Exception('this should never be called, but implemented by' 
        'the child class')

    def costs(self):
        '''lazy property to compute the costs per sample'''
        raise Exception('this should never be called, but implemented by' 
        'the child class')

    @lazy_property
    def cost(self):
        '''lazy property to compute the cost per batch'''
        loss =  tf.reduce_mean(self.costs)
        if self.use_tensorboard:
            tf.summary.scalar('loss', loss)
        return loss

    def scores(self):
        res = {}
        if self.l2:
            res['l2'] = tf.reduce_mean(tf.reduce_sum((self.ref - self.pred) ** 2, -1))
            if self.use_tensorboard:
                tf.summary.scalar('l2', res['l2'])
        return res


class ClassificationModel(Model):
    '''Abstract model class. '''
    def  __init__ (self,data,target,dropout,**kw):
        print("classificationmodel")
        super(ClassificationModel,self).__init__(data,target,dropout,**kw)
        self.target = target
        self.dropout = dropout
        self.learning_rate = argget(kw, 'learning_rate', 0.001)
        self.momentum = argget(kw, 'momentum', 0.9)
        self.nclasses = argget(kw, 'nclasses', 2)

        self.f05 = argget(kw, 'show_f05', True)
        self.f1 = argget(kw, 'show_f1', True)
        self.f2 = argget(kw, 'show_f2', True)

        self.dice = argget(kw, 'show_dice', not self.f1)
        
        self.cross_entropy = argget(kw, 'show_cross_entropy_loss', True)
        self.binary_evaluation = self.dice or self.f1 or self.f05 or self.f2
        self.eps = 1e-15
        fullscoreshape = [None for _ in self.target.get_shape()[:-1]]+[np.sum(self.nclasses)]
        fullscoreshape_minimum = [1 if s is None else s for s in fullscoreshape]
        self.ref = tf.placeholder_with_default(np.zeros((fullscoreshape_minimum),dtype=np.float32),fullscoreshape)
        self.pred = tf.placeholder_with_default(np.zeros((fullscoreshape_minimum),dtype=np.float32),fullscoreshape)

    def scores(self):
        with tf.device('/cpu:0'):
            res = super(ClassificationModel,self).scores()
            if self.binary_evaluation:
                enc_ref = tf.arg_max(self.ref,-1)
                enc_pred = self.nclasses*tf.arg_max(self.pred,-1)
                enc_both = enc_ref+enc_pred
                bins = tf.bincount(enc_both,minlength=self.nclasses**2,maxlength=self.nclasses**2).reshape((self.nclasses,self.nclasses))
            if self.dice:
                res['dice'] = [bins[c,c]*2/(tf.reduce_sum(bins,-1)[c]+tf.reduce_sum(bins,-2)[c]) for c in range(self.nclasses)]
                if self.use_tensorboard:
                    [tf.summary.scalar('dice-{}'.format(c),res['dice'][c]) for c in range(self.nclasses)]
            if self.f1:
                res['f1'] = [bins[c,c]*2/(tf.reduce_sum(bins,-2)[c]+tf.reduce_sum(bins,-1)[c]) for c in range(self.nclasses)]
                if self.use_tensorboard:
                    [tf.summary.scalar('f1-{}'.format(c),res['f1'][c]) for c in range(self.nclasses)]
            if self.cross_entropy:
                res['cross_entropy'] = tf.reduce_mean(tf.reduce_sum(self.ref*tf.log(self.pred+self.eps),-1))
                if self.use_tensorboard:
                    tf.summary.scalar('cross-entropy',res['cross_entropy'])
            return res

    def scores_np(self,ref,pred):
        res = {}
        eps = 1e-8
        if self.binary_evaluation:
            enc_ref = np.argmax(ref,-1)
            enc_pred = self.nclasses*np.argmax(pred,-1)
            enc_both = enc_ref+enc_pred
            bins = np.bincount(enc_both.flatten(), minlength=self.nclasses**2).reshape((self.nclasses, self.nclasses))
        if self.dice:
            res['dice'] = [bins[c, c] * 2 / (np.sum(bins,-1)[c] + np.sum(bins,-2)[c] + eps) for c in range(self.nclasses)]
        if self.f05 or self.f2:
            precision = np.array([bins[c, c] / (np.sum(bins, -1)[c] + eps) for c in range(self.nclasses)])
            recall = np.array([bins[c, c] / (np.sum(bins, -2)[c] + eps) for c in range(self.nclasses)])
        if self.f05:
            beta2 = 0.5**2
            res['f05'] = (1 + beta2) * precision * recall / ((beta2 * precision) + recall + eps)
        if self.f1:
            res['f1'] = [bins[c,c] * 2 / (np.sum(bins,-2)[c] + np.sum(bins,-1)[c] + eps) for c in range(self.nclasses)]
        if self.f2:
            beta2 = 2**2
            res['f2'] = (1 + beta2) * precision * recall / (beta2 * precision + recall + eps)
        if self.cross_entropy:
            res['cross_entropy'] = np.mean(np.sum(ref*np.log(pred+self.eps),-1))
        if self.l2:
            res['l2'] = np.mean(np.sum((ref-pred)**2,-1))
        return res


class RegressionModel(Model):
    """Abstract model class.
    """
    def __init__(self, data, target, dropout, **kw):
        super(RegressionModel, self).__init__(data, target, dropout, **kw)
        self.target = target
        self.dropout = dropout
        self.learning_rate = argget(kw, 'learning_rate', 0.001)
        self.nclasses = argget(kw, 'nclasses', 1)
        self.momentum = argget(kw, 'momentum', 0.9)


class ReconstructionModel(Model):
    """Abstract model class.
    """
    def __init__(self, data, dropout, **kw):
        super(ReconstructionModel, self).__init__(data, **kw)
        self.dropout = dropout
        self.learning_rate = argget(kw, 'learning_rate', 0.001)
        self.nclasses = argget(kw, 'nclasses', 2)


class GANModel(Model):
    def __init__(self, data, dropout, **kw):
        super(GANModel, self).__init__(data, dropout, **kw)
        self.dropout = dropout
        self.learning_rate = argget(kw, 'learning_rate', 0.001)
        self.momentum = argget(kw, 'momentum', 0.9)
        self.nclasses = argget(kw, 'nclasses', 2)
        self.fakedata = argget(kw, 'fakedata', None)
