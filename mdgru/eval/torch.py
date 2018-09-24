from eval import SupervisedEvaluation
import logging
import os
from helper import argget, check_if_kw_empty
import torch as th
import numpy as np
import copy
import time
from torch.autograd import Variable
from helper import compile_arguments

class SupervisedEvaluationTorch(SupervisedEvaluation):
    '''Base class for all evaluation classes. Child classes implement various
        test_* methods to test modeldependent aspects.

    Attributes:
        sess: tensorflow session. contains all the model data.
        saver: tensorflow saver to save or load the model data.

    '''
    _defaults = {}
    def __init__(self, modelcls, collectioninst, kw):
        super(SupervisedEvaluationTorch, self).__init__(modelcls, collectioninst, kw)
        eval_kw, kw = compile_arguments(SupervisedEvaluationTorch, kw, transitive=False)
        for k, v in eval_kw.items():
            setattr(self, k, v)
        data_shape = self.trdc.get_shape()
        kw['nclasses'] = self.output_dims
        modelkw, kw = compile_arguments(modelcls, kw, True, keep_entries=True)
        self.model = modelcls(data_shape, self.dropout_rate, modelkw)
        self.model.initialize()
        if len(self.gpu):
            self.model.model.cuda(self.gpu[0])
        self.optimizer = th.optim.Adadelta(self.model.model.parameters(), lr=self.model.learning_rate,
                                           rho=self.model.momentum)

        self.input_shape = [1] + data_shape[1:]
        self.batch = th.FloatTensor(*self.input_shape)
        self.batchlabs = th.LongTensor(*self.input_shape)
        if len(self.gpu):
            self.batch = self.batch.cuda(self.gpu[0])
            self.batchlabs = self.batchlabs.cuda(self.gpu[0])

        # check_if_kw_empty(self.__class__.__name__, kw, 'eval')

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
        self.model.train(True)
        losses = self.model.losses(self.model.model(self.batch), self.batchlabs)
        loss = 0
        for l in losses:
            loss+=l
        loss.backward()
        self.optimizer.step()
        return [loss.item() for loss in losses]


    def _predict_with_loss(self, batch, batchlabs):
        """run evaluation and calculate loss"""
        self.check_input(batch, batchlabs)
        self.model.train(False)
        result = self.model.model(self.batch)
        prediction = th.nn.softmax(result)
        return [loss.item() for loss in self.model.losses(result, self.batchlabs)], prediction.data.cpu().numpy()



    def _predict(self, batch, dropout, testing):
        """ predict given our graph for batch. Be careful as this method returns results always in NHWC or NDHWC"""
        batch_shape = batch.shape
        reorder = [0] + [i for i in range(2, len(batch_shape))] + [1]
        self.check_input(batch)
        self.model.train(False)
        return self.model.prediction(self.batch).data.cpu().numpy().transpose(reorder)


    def get_globalstep(self):
        return next(iter(self.optimizer.state_dict().values()))['step']

    def _save(self, f):
        """Save model"""
        modelstate = self.model.state_dict()
        optimizerstate = self.optimizer.state_dict()
        globalstep = next(iter(optimizerstate['state'].values()))['step']
        th.save({'model': modelstate, 'optimizer': optimizerstate}, f + "-{}".format(globalstep))
        return f + '-{}'.format(globalstep)

    def _load(self, f):
        """Load model"""
        state = th.load(f)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

