__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import copy
import logging
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from helper import argget

try:
    import cPickle as pickle
except:
    import pickle


class Evaluation(object):
    estimatefilename = "estimate.nii"
    '''Base class for all evaluation classes. Child classes implement various 
        test_* methods to test modeldependent aspects.
    
    Attributes:
        sess: tensorflow session. contains all the model data. 
        saver: tensorflow saver to save or load the model data.
    
    '''

    def __init__(self, collectioninst, **kw):
        self.origargs = copy.deepcopy(kw)
        self.use_tensorboard = argget(kw, "use_tensorboard", True, keep=True)
        if self.use_tensorboard:
            self.image_summaries_each = argget(kw, 'image_summaries_each', 100)
        self.dropout_rate = argget(kw, "dropout_rate", 0.5)
        self.current_epoch = 0
        self.current_iteration = 0
        self.restore_optimistically = argget(kw, 'restore_optimistically', False)
        # init tf in runner, but set sell.sess as session!

    def set_session(self, sess, cachefolder):
        self.sess = sess
        sess.run(tf.global_variables_initializer())
        if self.use_tensorboard:
            self.train_writer = tf.summary.FileWriter(os.path.join(cachefolder, 'train'), sess.graph)
            self.train_image_writer = tf.summary.FileWriter(os.path.join(cachefolder, 'train_imgs'), sess.graph)
            self.validation_writer = tf.summary.FileWriter(os.path.join(cachefolder, 'validation'), sess.graph)
            self.test_writer = tf.summary.FileWriter(os.path.join(cachefolder, 'test'))
            self.evaluate_merged = True
            self.merged_only_images = tf.summary.merge_all(key='images')
            self.merged_basic = tf.summary.merge_all()
            self.merged_images = tf.summary.merge([self.merged_basic, self.merged_only_images])
        else:
            self.evaluate_merged = False

        logging.getLogger('eval').info('initialized all variables')
        self.saver = tf.train.Saver(max_to_keep=None)

    def train(self, batch_size):
        '''Method to evaluate batch_size number of samples and adjust weights 
            once.
        
        Args:
            batch_size (int): number of samples to evaluate before adjusting 
                the gradient.
                
        Returns:
            The evaluated model loss.
        '''
        raise Exception('abstract method')

    def save(self, f):
        '''saves model to disk at location f'''
        self.saver.save(self.sess, f, global_step=self.model.global_step)
        trdc = self.trdc.getStates()
        tedc = self.tedc.getStates()
        valdc = self.valdc.getStates()
        states = {}
        if trdc:
            states['trdc'] = trdc
        if tedc:
            states['tedc'] = tedc
        if valdc:
            states['valdc'] = valdc
        states['epoch'] = self.current_epoch
        states['iteration'] = self.current_iteration
        pickle.dump(states, open(f + ".pickle", "wb"))

    def optimistic_restore(self, session, save_file):
        reader = tf.train.NewCheckpointReader(save_file)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                            if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        name2var = dict(zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                curr_var = name2var[saved_var_name]
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(curr_var)
        saver = tf.train.Saver(restore_vars)
        saver.restore(session, save_file)

    def load(self, f):
        '''loads model at location f from disk'''
        if self.restore_optimistically:
            self.optimistic_restore(self.sess, f)
        else:
            try:
                self.saver.restore(self.sess, f)
            except Exception as e:
                try:
                    reader = pywrap_tensorflow.NewCheckpointReader(f)
                    var_to_shape_map = reader.get_variable_to_shape_map()
                    tensor_names = []
                    for i, key in enumerate(sorted(var_to_shape_map)):
                        tensor_names.append(key)
                        if i == 10:
                            break
                    logging.getLogger('eval').warning(
                            'the following are the first tensor_names in checkpoint file {}: {}'
                            .format(f, ",".join(tensor_names)))
                finally:
                    raise e

        states = {}
        try:
            pickle_name = f.rsplit('-', 1)[0] + ".pickle"
            states = pickle.load(open(pickle_name, "rb"))
        except Exception as e:
            logging.getLogger('eval').warning('there was no randomstate pickle named {} around'.format(pickle_name))
        if "trdc" in states:
            self.trdc.setStates(states['trdc'])
        else:
            self.trdc.setStates(None)
        if "tedc" in states:
            self.tedc.setStates(states['tedc'])
        else:
            self.tedc.setStates(None)
        if "valdc" in states:
            self.valdc.setStates(states['valdc'])
        else:
            self.valdc.setStates(None)
        if 'epoch' in states:
            self.current_epoch = states['epoch']
        else:
            self.current_epoch = 0
        if 'iteration' in states:
            self.current_iteration = states['iteration']
        else:
            self.current_iteration = 0
        self.current_iteration = self.current_iteration

    def test_all_random(self, **kw):
        '''Method to test, depends on target of course'''
        raise Exception('abstract method')

    def test_scores(self, pred=None, tar=None):
        res = None
        raise Exception('this has to be implemented separately!')
        return res

    def setupCollections(self, collectioninst):
        if isinstance(collectioninst, dict):
            self.trdc = collectioninst['train']
            self.tedc = collectioninst['test']
            self.dcs = collectioninst
            if "validation" in collectioninst:
                self.valdc = collectioninst['validation']
                logging.getLogger('eval').debug('using validation valdc')
            else:
                self.valdc = self.tedc

        else:
            self.trdc = collectioninst
            self.dcs = {'train': collectioninst}
            self.tedc = collectioninst
            self.valdc = collectioninst


class SupervisedEvaluation(Evaluation):
    def __init__(self, model, collectioninst, **kw):
        super(SupervisedEvaluation, self).__init__(collectioninst, **kw)
        self.setupCollections(collectioninst)
        self.currit = 0
        self.namespace = argget(kw, "namespace", "default")
        with tf.variable_scope(self.namespace):
            self.training = tf.placeholder(dtype=tf.bool)
            self.dropout = tf.placeholder(dtype=tf.float32)
            self.data = tf.placeholder(dtype=tf.float32, shape=self.trdc.get_shape())
            if type(self.nclasses) == list:  # for location classification
                self.target = tf.placeholder(dtype=tf.float32,
                                             shape=self.trdc.get_target_shape()[:-1] + [np.sum(self.nclasses)])
            else:
                self.target = tf.placeholder(dtype=tf.float32,
                                             shape=self.trdc.get_target_shape()[:-1] + [self.nclasses])
            kw_copy = copy.deepcopy(kw)
            self.model = model(self.data, self.target, self.dropout, training=self.training, **kw)
            self.model.optimize
# in the case we have a different testing set, we can construct 2 graphs, one for training and testing case
        if self.tedc.get_shape() != self.trdc.get_shape():
            self.test_graph = tf.Graph()
            with self.test_graph.as_default():
                with tf.variable_scope(self.namespace):
                    self.test_training = tf.placeholder(dtype=tf.bool)
                    self.test_dropout = tf.placeholder(dtype=tf.float32)
                    self.test_data = tf.placeholder(dtype=tf.float32, shape=self.tedc.get_shape())
                    if type(self.nclasses) == list:  # for location classification
                        self.test_target = tf.placeholder(dtype=tf.float32,
                                                     shape=self.tedc.get_target_shape()[:-1] + [np.sum(self.nclasses)])
                    else:
                        self.test_target = tf.placeholder(dtype=tf.float32,
                                                     shape=self.tedc.get_target_shape()[:-1] + [self.nclasses])
                    self.test_model = model(self.test_data, self.test_target, self.test_dropout, training=self.test_training,
                                            **kw_copy)
                    self.test_model.prediction
                    self.test_model.cost
        else:
            self.test_graph = tf.get_default_graph()
            self.test_model = self.model
            self.test_training = self.training
            self.test_dropout = self.dropout
            self.test_data = self.data
            self.test_target = self.target

        self.batch_size = argget(kw, 'batch_size', 1)
        self.validate_same = argget(kw, 'validate_same', False)

    def train(self, batch_size=None):
        start_time = time.time()
        if batch_size is None:
            batch_size = self.batch_size
        batch, batchlabs = self.trdc.random_sample(batch_size=self.batch_size)
        time_after_loading = time.time()
        tasks = [self.model.optimize, self.model.cost]
        ph = {self.data: batch, self.target: batchlabs, self.dropout: self.dropout_rate, self.training: True}
        if self.evaluate_merged:
            if self.currit % self.image_summaries_each == 0:
                tasks.append(self.merged_images)
            else:
                tasks.append(self.merged_basic)
            _, loss, summary = self.sess.run(tasks, ph)
            self.train_writer.add_summary(summary, tf.train.get_global_step())
        else:
            _, loss = self.sess.run(tasks, ph)
        self.currit += 1
        end_time = time.time()
        logging.getLogger('eval').info("it: {}, time: [i/o: {}, processing: {}, all: {}], loss: {}"
                                       .format(self.currit,
                                               np.round(time_after_loading - start_time, 6),
                                               np.round(end_time - time_after_loading, 6),
                                               np.round(end_time - start_time,6),
                                               loss))
        return loss

    def test_loss_random(self, batch_size=None, dc=None, resample=True):
        if dc is None:
            dc = self.valdc
        if batch_size is None:
            batch_size = self.batch_size
        if resample:
            self.testbatch, self.testbatchlabs = dc.random_sample(batch_size=self.batch_size)
        return self.sess.run(self.model.costs,
                             {self.data: self.testbatch, self.target: self.testbatchlabs, self.dropout: 1,
                              self.training: False})

    def test_pred_random(self, batch_size=None, dc=None, resample=True):
        if dc is None:
            dc = self.valdc
        if batch_size is None:
            batch_size = self.batch_size
        if resample:
            self.testbatch, self.testbatchlabs = dc.random_sample(batch_size=self.batch_size)
        return self.sess.run(self.model.prediction,
                             {self.data: self.testbatch, self.dropout: 1, self.training: False})

    def test_scores(self, pred, tar):
        tar = np.int32(np.expand_dims(tar.squeeze(), 0))
        pred = np.expand_dims(pred.squeeze(), 0)
        if pred.shape != tar.shape:
            tar2 = np.zeros((np.prod(pred.shape[:-1]), pred.shape[-1]))
            tar2[np.arange(np.prod(pred.shape[:-1])), tar.flatten()] = 1
            tar = tar2.reshape(pred.shape)
        res = self.model.scores_np(tar, pred)
        return res

    def test_all_random(self, batch_size=None, dc=None, resample=True, summary_writer=None):
        if dc is None:
            dc = self.valdc
        if batch_size is None:
            batch_size = self.batch_size
        if self.validate_same:
            dc.randomstate.seed(12345677)
        if resample:
            self.testbatch, self.testbatchlabs = dc.random_sample(batch_size=batch_size)
        tasks = [self.model.costs, self.model.prediction]
        placeholders = {self.data: self.testbatch, self.target: self.testbatchlabs, self.dropout: 1,
                        self.training: False}
        if self.evaluate_merged:
            if summary_writer is None:
                summary_writer = self.test_writer
            tasks.append(self.merged_basic)
            b, c, summary = self.sess.run(tasks, placeholders)
            summary_writer.add_summary(summary)
        else:
            b, c = self.sess.run(tasks, placeholders)
        return b, c

    def test_all_available(self, batch_size=None, dc=None):
        if dc is None:
            dc = self.tedc
        raise Exception('this should be implemented')


class LargeVolumeEvaluation(Evaluation):
    def __init__(self, model, collectioninst, **kw):
        self.evaluate_uncertainty_times = argget(kw, "evaluate_uncertainty_times", 1)
        self.evaluate_uncertainty_dropout = argget(kw, "evaluate_uncertainty_dropout",
                                                   1.0)  # these standard values ensure that we dont evaluate uncertainty if nothing was provided.
        self.evaluate_uncertainty_saveall = argget(kw, "evaluate_uncertainty_saveall", False)
        super(LargeVolumeEvaluation, self).__init__(model, collectioninst, **kw)

    def test_all_available(self, batch_size=None, dc=None, return_results=False, dropout=None, testing=False):
        if dc is None:
            dc = self.tedc
        if testing:
            model = self.test_model
            data = self.test_data
            dropoutph = self.test_dropout
            trainingph = self.test_training
        else:
            model = self.model
            data = self.data
            trainingph = self.training
            dropoutph = self.dropout
        if batch_size > 1:
            logging.getLogger('eval').error('not supported yet to have more than batchsize 1')
        volgens = dc.get_volume_batch_generators()
        if dropout is None:
            dropout = self.evaluate_uncertainty_dropout
        full_vols = []
        errs = []

        lasttime = time.time()
        for volgen, file, shape, w, p in volgens:
            logging.getLogger('eval').info(
                'evaluating file {} of shape {} with w {} and p {}'.format(file, shape, w, p))
            if len(shape) > 3:
                shape = np.asarray([s for s in shape if s > 1])
            res = np.zeros(list(shape) + [self.nclasses])
            if self.evaluate_uncertainty_times > 1:
                uncertres = np.zeros(res.shape)
                if self.evaluate_uncertainty_saveall:
                    allres = np.zeros([self.evaluate_uncertainty_times] + list(res.shape))
            certainty = np.ones(w)
            for ind, pp in enumerate(p):
                if pp > 0:
                    slicesa = [slice(None) for _ in range(len(p))]
                    slicesb = [slice(None) for _ in range(len(p))]
                    reshapearr = [1 for _ in range(len(p))]
                    reshapearr[ind] = pp
                    slicesa[ind] = slice(None, pp)
                    slicesb[ind] = slice(-pp, None)
                    certainty[slicesa] *= np.arange(1.0 / (pp + 1), 1, 1.0 / (pp + 1)).reshape(reshapearr)
                    certainty[slicesb] *= np.arange(1.0 / (pp + 1), 1, 1.0 / (pp + 1))[::-1].reshape(reshapearr)

            certainty = certainty.reshape([1] + w + [1])
            # read, compute, merge, write back
            for subvol, _, imin, imax in volgen:
                if self.evaluate_uncertainty_times > 1:
                    preds = []
                    for i in range(self.evaluate_uncertainty_times):
                        preds.append(self.sess.run(model.prediction,
                                                   {data: subvol, dropoutph: dropout, trainingph: False}))
                        logging.getLogger('eval').debug(
                            'evaluated run {} of subvolume from {} to {}'.format(i, imin, imax))
                    pred = np.mean(np.asarray(preds), 0)
                    uncert = np.std(np.asarray(preds), 0)
                    preds = [x * certainty for x in preds]
                else:
                    pred = self.sess.run(model.prediction,
                                         {data: subvol, dropoutph: dropout, trainingph: False})
                    logging.getLogger('eval').debug('evaluated subvolume from {} to {}'.format(imin, imax))

                pred *= certainty
                # now reembed it into array
                wrongmin = [int(abs(x)) if x < 0 else 0 for x in imin]
                wrongmax = [int(x) if x < 0 else None for x in (shape - imax)]
                mimin = np.asarray(np.maximum(0, imin), dtype=np.int32)
                mimax = np.asarray(np.minimum(shape, imax), dtype=np.int32)
                slicesaa = [slice(mimina, miminb) for mimina, miminb in zip(mimin, mimax)]
                slicesaa.append(slice(None))
                slicesbb = [0]
                slicesbb.extend(slice(wrongmina, wrongminb) for wrongmina, wrongminb in zip(wrongmin, wrongmax))
                slicesbb.append(slice(None))

                res[slicesaa] += pred[slicesbb]
                if self.evaluate_uncertainty_times > 1:
                    uncert *= certainty
                    uncertres[mimin[0]:mimax[0], mimin[1]:mimax[1], mimin[2]:mimax[2], :] += \
                        uncert[0, wrongmin[0]:wrongmax[0], wrongmin[1]:wrongmax[1], wrongmin[2]:wrongmax[2], :]
                    if self.evaluate_uncertainty_saveall:
                        for j in range(self.evaluate_uncertainty_times):
                            allres[j, mimin[0]:mimax[0], mimin[1]:mimax[1], mimin[2]:mimax[2], :] += \
                                preds[i][0, wrongmin[0]:wrongmax[0], wrongmin[1]:wrongmax[1], wrongmin[2]:wrongmax[2],
                                :]

            # normalize again:
            if self.evaluate_uncertainty_times > 1:
                uncertres /= np.sum(res, -1).reshape(list(res.shape[:-1]) + [1])
                dc.save(uncertres, os.path.join(file, "std-" + self.estimatefilename))
                if self.evaluate_uncertainty_saveall:
                    for j in range(self.evaluate_uncertainty_times):
                        dc.save(allres[j], os.path.join(file, "iter{}-".format(j) + self.estimatefilename))
            res /= np.sum(res, -1).reshape(list(res.shape[:-1]) + [1])
            # evaluate accuracy...
            name = os.path.split(file)
            name = name[-1] if len(name[-1]) else os.path.basename(name[0])
            try:
                if len(dc.maskfiles) > 0:
                    mfile = os.path.join(file, dc.maskfiles[0])
                    if os.path.exists(mfile):
                        mf = np.expand_dims(dc.load(mfile).squeeze(), 0)
                        errs.append([name, self.test_scores(res, mf)])
            except Exception as e:
                logging.getLogger('eval').warning('was not able to save test scores, even though ground truth was available.')
                logging.getLogger('eval').warning('{}'.format(e))
            if return_results:
                full_vols.append([name, file, res])
            else:
                dc.save(res, os.path.join(file, self.estimatefilename), tporigin=file)
            logging.getLogger('eval').info('evaluation took {} seconds'.format(time.time() - lasttime))
            lasttime = time.time()
        if return_results:
            return full_vols, errs
        else:
            return errs
