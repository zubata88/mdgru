from eval import SupervisedEvaluation
import tensorflow as tf
import logging
import os
import pickle
from tensorflow.python import pywrap_tensorflow
from helper import argget, check_if_kw_empty
import numpy as np
import copy

class SupervisedEvaluationTensorflow(SupervisedEvaluation):
    '''Base class for all evaluation classes. Child classes implement various
        test_* methods to test modeldependent aspects.
    
    Attributes:
        sess: tensorflow session. contains all the model data. 
        saver: tensorflow saver to save or load the model data.
    
    '''
    def __init__(self, model, collectioninst, kw):
        super(SupervisedEvaluationTensorflow, self).__init__(model, collectioninst, kw)
        self.use_tensorboard = argget(kw, "use_tensorboard", True, keep=True)
        if self.use_tensorboard:
            self.image_summaries_each = argget(kw, 'image_summaries_each', 100)

        self.restore_optimistically = argget(kw, 'restore_optimistically', False)

        self.only_cpu = argget(kw, 'only_cpu', False)
        self.gpubound = argget(kw, 'gpubound', 1)
        if self.only_cpu:
            self.session_config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            if self.gpubound < 1:
                self.session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=self.gpubound))
            else:
                self.session_config = tf.ConfigProto()

        with tf.variable_scope(self.namespace):
            self.training = tf.placeholder(dtype=tf.bool)
            self.dropout = tf.placeholder(dtype=tf.float32)
            self.data = tf.placeholder(dtype=tf.float32, shape=self.trdc.get_shape())
            if type(self.nclasses) == list:  # for location classification
                self.target = tf.placeholder(dtype=tf.float32,
                                             shape=self.trdc.get_target_shape()[:-1] + [np.sum(self.nclasses)])
            else:
                self.target = tf.placeholder(dtype=tf.float32,
                                             shape=self.trdc.get_target_shape())
            kw_copy = copy.deepcopy(kw)
            kw['training'] = self.training
            self.model = model(self.data, self.target, self.dropout, kw)
            self.model.optimize
        self.saver = tf.train.Saver(max_to_keep=None)
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
                                                          shape=self.tedc.get_target_shape())
                    kw_copy['test_training'] = self.test_training
                    self.test_model = model(self.test_data, self.test_target, self.test_dropout, kw_copy)
                    self.test_model.prediction
                    self.test_model.cost
                self.test_saver = tf.train.Saver(max_to_keep=None)
        else:
            self.test_graph = tf.get_default_graph()
            self.test_model = self.model
            self.test_training = self.training
            self.test_dropout = self.dropout
            self.test_data = self.data
            self.test_target = self.target
            self.test_saver = self.saver
        self.get_train_session = lambda: tf.Session(config=self.session_config)
        self.get_test_session = lambda: tf.Session(config=self.session_config, graph=self.test_graph)

        check_if_kw_empty(self.__class__.__name__, kw, 'eval')

    def _train(self, batch, batchlabs):
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
        return loss

    def _predict_with_loss(self, batch, batchlabs):
        tasks = [self.model.costs, self.model.prediction]
        placeholders = {self.data: batch, self.target: batchlabs, self.dropout: 1,
                        self.training: False}
        if self.evaluate_merged:
            summary_writer = self.test_writer
            tasks.append(self.merged_basic)
            loss, prediction, summary = self.sess.run(tasks, placeholders)
            summary_writer.add_summary(summary)
        else:
            loss, prediction = self.sess.run(tasks, placeholders)
        return loss, prediction

    def _predict(self, batch, dropout, testing):
        """ predict given our graph for batch."""
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
        return self.sess.run(model.prediction, {data: batch, dropoutph: dropout, trainingph: False})

    def set_session(self, sess, cachefolder, train=False):
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

    def get_globalstep(self):
        return self.sess.run(self.model.global_step)

    def _save(self, f):
        globalstep = self.get_globalstep()
        self.saver.save(self.sess, f, global_step=self.model.global_step)
        return f + '-{}'.format(globalstep) #checkpointfilename

    def _optimistic_restore(self, session, save_file):
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

    def _load(self, f):
        if self.restore_optimistically:
            self._optimistic_restore(self.sess, f)
        else:
            try:
                self.saver.restore(self.sess, f)
            except Exception as e:
                import traceback
                traceback.print_exc()
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

    def add_summary_simple_value(self, text, value):
        summary = tf.Summary()
        summary.value.add(tag=text, simple_value=value)
        self.train_writer.add_summary(summary)