from eval import SupervisedEvaluation
import logging
import os
from helper import argget, check_if_kw_empty
import torch as th
import numpy as np
import copy
import time
from torch.autograd import Variable

class SupervisedEvaluationTorch(SupervisedEvaluation):
    '''Base class for all evaluation classes. Child classes implement various
        test_* methods to test modeldependent aspects.

    Attributes:
        sess: tensorflow session. contains all the model data.
        saver: tensorflow saver to save or load the model data.

    '''
    def __init__(self, model, collectioninst, kw):
        super(SupervisedEvaluationTorch, self).__init__(model, collectioninst, kw)
        data_shape = self.trdc.get_shape()
        self.model = model(data_shape, self.dropout_rate, kw)
        self.model.initialize()
        if len(self.gpu):
            self.model.logits.cuda(self.gpu[0])
        self.optimizer = th.optim.Adadelta(self.model.logits.parameters(), lr=self.model.learning_rate, rho=self.model.momentum)
        # self.use_tensorboard = argget(kw, "use_tensorboard", True, keep=True)
        # if self.use_tensorboard:
        #     self.image_summaries_each = argget(kw, 'image_summaries_each', 100)

        # self.restore_optimistically = argget(kw, 'restore_optimistically', False)
        self.input_shape = [1] + data_shape[1:]
        self.batch = th.FloatTensor(*self.input_shape)
        self.batchlabs = th.LongTensor(*self.input_shape)
        if len(self.gpu):
            self.batch = self.batch.cuda(self.gpu[0])
            self.batchlabs = self.batchlabs.cuda(self.gpu[0])

        self.only_cpu = argget(kw, 'only_cpu', False)
        # self.gpubound = argget(kw, 'gpubound', 1)
        # if self.only_cpu:
        #     self.session_config = tf.ConfigProto(device_count={'GPU': 0})
        # else:
        #     if self.gpubound < 1:
        #         self.session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=self.gpubound))
        #     else:
        #         self.session_config = tf.ConfigProto()
        # raise Exception("not yet implemented")
        # with tf.variable_scope(self.namespace):
        #     self.training = tf.placeholder(dtype=tf.bool)
        #     self.dropout = tf.placeholder(dtype=tf.float32)
        #     self.data = tf.placeholder(dtype=tf.float32, shape=self.trdc.get_shape())
        #     if type(self.nclasses) == list:  # for location classification
        #         self.target = tf.placeholder(dtype=tf.float32,
        #                                      shape=self.trdc.get_target_shape()[:-1] + [np.sum(self.nclasses)])
        #     else:
        #         self.target = tf.placeholder(dtype=tf.float32,
        #                                      shape=self.trdc.get_target_shape()[:-1] + [self.nclasses])
        #     kw_copy = copy.deepcopy(kw)
        #     kw['training'] = self.training
        #     self.model = model(self.data, self.target, self.dropout, kw)
        #     self.model.optimize
        # in the case we have a different testing set, we can construct 2 graphs, one for training and testing case
        # if self.tedc.get_shape() != self.trdc.get_shape():
        #     self.test_graph = tf.Graph()
        #     with self.test_graph.as_default():
        #         with tf.variable_scope(self.namespace):
        #             self.test_training = tf.placeholder(dtype=tf.bool)
        #             self.test_dropout = tf.placeholder(dtype=tf.float32)
        #             self.test_data = tf.placeholder(dtype=tf.float32, shape=self.tedc.get_shape())
        #             if type(self.nclasses) == list:  # for location classification
        #                 self.test_target = tf.placeholder(dtype=tf.float32,
        #                                                   shape=self.tedc.get_target_shape()[:-1] + [np.sum(self.nclasses)])
        #             else:
        #                 self.test_target = tf.placeholder(dtype=tf.float32,
        #                                                   shape=self.tedc.get_target_shape()[:-1] + [self.nclasses])
        #             kw_copy['test_training'] = self.test_training
        #             self.test_model = model(self.test_data, self.test_target, self.test_dropout, kw_copy)
        #             self.test_model.prediction
        #             self.test_model.cost
        # else:
        #     self.test_graph = tf.get_default_graph()
        #     self.test_model = self.model
        #     self.test_training = self.training
        #     self.test_dropout = self.dropout
        #     self.test_data = self.data
        #     self.test_target = self.target

        check_if_kw_empty(self.__class__.__name__, kw, 'eval')

    def check_input(self, batch, batchlabs=None):
        batch = th.from_numpy(batch)
        if batchlabs is not None:
            batchlabs = th.from_numpy(batchlabs)
        if batch.shape != self.input_shape:
            self.input_shape = batch.shape
            self.batch.resize_(batch.size())
            if batchlabs is not None:
                self.batchlabs.resize_(batchlabs.size())
        self.batch.copy_(batch)
        if batchlabs is not None:
            self.batchlabs.copy_(batchlabs)

    def _train(self, batch, batchlabs):
        """set inputs and run torch training iteration"""
        self.check_input(batch, batchlabs)
        self.optimizer.zero_grad()
        loss = self.model.cost(self.model.logits(Variable(self.batch)), Variable(self.batchlabs))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # raise Exception("not yet implemented")
        # tasks = [self.model.optimize, self.model.cost]
        # ph = {self.data: batch, self.target: batchlabs, self.dropout: self.dropout_rate, self.training: True}
        # if self.evaluate_merged:
        #     if self.currit % self.image_summaries_each == 0:
        #         tasks.append(self.merged_images)
        #     else:
        #         tasks.append(self.merged_basic)
        #     _, loss, summary = self.sess.run(tasks, ph)
        #     self.train_writer.add_summary(summary, tf.train.get_global_step())
        # else:
        #     _, loss = self.sess.run(tasks, ph)
        # return loss

    def _predict_with_loss(self, batch, batchlabs):
        """run evaluation and calculate loss"""
        self.check_input(batch, batchlabs)
        logits = self.model.logits(self.batch)
        prediction = th.nn.softmax(logits)
        return self.model.cost(logits, self.batchlabs).data[0], prediction.data.cpu().numpy()

        # raise Exception("not yet implemented")
        # tasks = [self.model.costs, self.model.prediction]
        # placeholders = {self.data: batch, self.target: batchlabs, self.dropout: 1,
        #                 self.training: False}
        # if self.evaluate_merged:
        #     summary_writer = self.test_writer
        #     tasks.append(self.merged_basic)
        #     loss, prediction, summary = self.sess.run(tasks, placeholders)
        #     summary_writer.add_summary(summary)
        # else:
        #     loss, prediction = self.sess.run(tasks, placeholders)
        # return loss, prediction

    def _predict(self, batch, dropout, testing):
        """ predict given our graph for batch. Be careful as this method returns results always in NHWC or NDHWC"""
        batch_shape = batch.shape
        reorder = [0] + [i for i in range(2, len(batch_shape))] + [1]
        self.check_input(batch)
        return th.nn.functional.softmax(self.model.logits(self.batch)).data.cpu().numpy().transpose(reorder)
        # raise Exception("not yet implemented")
        # if testing:
        #     model = self.test_model
        #     data = self.test_data
        #     dropoutph = self.test_dropout
        #     trainingph = self.test_training
        # else:
        #     model = self.model
        #     data = self.data
        #     trainingph = self.training
        #     dropoutph = self.dropout
        # return self.sess.run(model.prediction, {data: batch, dropoutph: dropout, trainingph: False})

    def get_globalstep(self):
        return next(iter(self.optimizer.state_dict().values()))['step']

    def _save(self, f):
        """Save model"""
        modelstate = self.model.state_dict()
        optimizerstate = self.optimizer.state_dict()
        globalstep = next(iter(optimizerstate['state'].values()))['step']
        th.save({'model': modelstate, 'optimizer': optimizerstate}, f + "-{}".format(globalstep))
        return f + '-{}'.format(globalstep) 
        # raise Exception("not yet implemented")
        # self.saver.save(self.sess, f, global_step=self.model.global_step)

    def _load(self, f):
        """Load model"""
        state = th.load(f)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        # raise Exception("not yet implemented")
        # if self.restore_optimistically:
        #     self._optimistic_restore(self.sess, f)
        # else:
        #     try:
        #         self.saver.restore(self.sess, f)
        #     except Exception as e:
        #         try:
        #             reader = pywrap_tensorflow.NewCheckpointReader(f)
        #             var_to_shape_map = reader.get_variable_to_shape_map()
        #             tensor_names = []
        #             for i, key in enumerate(sorted(var_to_shape_map)):
        #                 tensor_names.append(key)
        #                 if i == 10:
        #                     break
        #             logging.getLogger('eval').warning(
        #                 'the following are the first tensor_names in checkpoint file {}: {}'
        #                     .format(f, ",".join(tensor_names)))
        #         finally:
        #             raise e
        #

