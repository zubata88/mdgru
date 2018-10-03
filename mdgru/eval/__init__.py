__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import copy
import logging
import os
import pickle
import time
from abc import abstractmethod

import numpy as np

from mdgru.helper import argget, compile_arguments, generate_defaults_info


class SupervisedEvaluation(object):

    _defaults = {
        'dropout_rate': {'value': 0.5,
                         'help': '"keep rate" for weights using dropconnect. The higher the value, the closer the sampled models to the full model.'},
        'namespace': {'value': 'default',
                      'help': "override default model name (if no ckpt is provided). Probably not a good idea!",
                      'alt': ['modelname']},
        'only_save_labels': {'value': False, 'help': 'save only labels and no probability distributions'},
        # 'batch_size': {'value': 1, 'help': 'determines batch size to be used during training'}
        'validate_same': {'value': True, 'help': 'always pick other random samples for validation!',
                          'invert_meaning': 'dont_'},
        'evaluate_uncertainty_times': {'value': 1, 'type': int,
                                       'help': 'Number times we want to evaluate one volume. This only makes sense '
                                               'using a keep rate of less than 1 during evaluation (dropout_during_evaluation '
                                               'less than 1)', 'name': 'number_of_evaluation_samples'},
        'evaluate_uncertainty_dropout': {'value': 1.0, 'type': float,
                                         'help': 'Keeprate of weights during evaluation. Useful to visualize uncertainty '
                                                 'in conjunction with a number of samples per volume',
                                         'name': 'dropout_during_evaluation'},
        'evaluate_uncertainty_saveall': {'value': False,
                                         'help': 'Save each evaluation sample per volume. Without this flag, only the '
                                                 'standard deviation and mean over all samples is kept.',
                                         'name': 'save_individual_evaluations'},
        'show_f05': True,
        'show_f1': True,
        'show_f2': True,
        'show_l2': True,
        'show_cross_entropy': True,
        'print_each': {'value': 1, 'help': 'print execution time and losses each # iterations', 'type': int},
        'batch_size': {'value': 1, 'help': 'Minibatchsize', 'type': int, 'name': 'batchsize', 'short': 'b'},
        'datapath': {
            'help': 'path where training, validation and testing folders lie. Can also be some other path, as long as the other locations are provided as absolute paths. An experimentsfolder will be created in this folder, where all runs and checkpoint files will be saved.'},
        'locationtraining': {'value': None,
                             'help': 'absolute or relative path to datapath to the training data. Either a list of paths to the sample folders or one path to a folder where samples should be automatically determined.',
                             'nargs': '+'},
        'locationtesting': {'value': None,
                            'help': 'absolute or relative path to datapath to the testing data. Either a list of paths to the sample folders or one path to a folder where samples should be automatically determined.',
                            'nargs': '+'},
        'locationvalidation': {'value': None,
                               'help': 'absolute or relative path to datapath to the validation data. Either a list of paths to the sample folders or one path to a folder where samples should be automatically determined.',
                               'nargs': '+'},
        'output_dims': {'help': 'number of output channels, e.g. number of classes the model needs to create a probability distribution over.','type': int, 'alt': ['nclasses']},
        'windowsize': {'type':int, 'short':'w','help': 'window size to be used during training, validation and testing, if not specified otherwise', 'nargs':'+'},
        'padding': {'help': 'padding to be used during training, validation and testing, if not specified otherwise. During training, the padding specifies the amount a patch is allowed to reach outside of the image along all dimensions, during testing, it specifies also the amount of overlap needed between patches.', 'value': [0], 'nargs':'+', 'short':'p', 'type':int},
        'windowsizetesting': {'value':None, 'help': 'override windowsize for testing','nargs':'+', 'type':int},
        'windowsizevalidation': None,#{'value':None, 'help': 'override windowsize for validation','nargs':'+'},
        'paddingtesting': {'value':None, 'help': 'override padding for testing', 'nargs':'+', 'type':int},
        'paddingvalidation': None,#{'value':None, 'help': 'override padding for validation', 'nargs':'+'},
        'testbatchsize': {'value': 1, 'help': 'batchsize for testing'}
    }

    def __init__(self, modelcls, datacls, kw):
        """
        Handler for the evaluation of model defined in modelcls using data coming from datacls.

        Parameters
        ----------
        modelcls : cls
            Python class defining the model to evaluate
        datacls : cls
            Python class implementing the data loading and storing

        """
        self.origargs = copy.copy(kw)
        eval_kw, kw = compile_arguments(SupervisedEvaluation, kw, transitive=False)
        for k, v in eval_kw.items():
            setattr(self, k, v)
        self.w = self.windowsize
        self.p = self.padding
        self.use_tensorboard = False
        # self.dropout_rate = argget(kw, "dropout_rate", 0.5)
        self.current_epoch = 0
        self.current_iteration = 0
        # create datasets for training, validation and testing:
        locs = [[None, l] if l is None or len(l) > 1 else [os.path.join(self.datapath, l[0]), None] for l in
                [self.locationtraining, self.locationvalidation, self.locationtesting]]
        paramstraining = [self.w, self.p] + locs[0]
        paramsvalidation = [self.windowsizevalidation if self.windowsizevalidation is not None else self.w,
                            self.paddingvalidation if self.paddingvalidation is not None else self.p] + locs[1]
        paramstesting = [self.windowsizetesting if self.windowsizetesting is not None else self.w,
                            self.paddingtesting if self.paddingtesting is not None else self.p] + locs[2]
        kwdata, kw = compile_arguments(datacls, kw, True, keep_entries=True)
        kwcopy = copy.copy(kwdata)
        kwcopy['nclasses'] = self.output_dims
        kwcopy['batch_size'] = self.batch_size
        self.trdc = datacls(*paramstraining, kw=copy.copy(kwcopy))
        testkw = copy.copy(kwcopy)
        testkw['batch_size'] = testkw['batch_size'] if not self.testbatchsize else self.testbatchsize
        valkw = copy.copy(testkw)
        testkw['ignore_missing_mask'] = True
        self.tedc = datacls(*paramstesting, kw=testkw)
        self.valdc = datacls(*paramsvalidation, kw=valkw)
        self.currit = 0

        self.show_dice = argget(kw, "show_dice", not self.show_f1)
        self.binary_evaluation = self.show_dice or self.show_f1 or self.show_f05 or self.show_f2
        self.estimatefilename = argget(kw, "estimatefilename", "estimate")
        self.gpu = argget(kw, "gpus", [0])
        self.get_train_session = lambda: self
        self.get_test_session = lambda: self

    @abstractmethod
    def _train(self):
        """ Performs one training iteration in respective framework and returns loss(es)"""
        raise Exception("This needs to be implemented depending on the framework")

    @abstractmethod
    def _predict(self, batch, dropout, testing):
        """
        Predict given batch and keeprate dropout.

        Parameters
        ----------
        batch : ndarray
        dropout : float
            Keeprate for dropconnect
        testing

        Returns
        -------
        ndarray : Prediction based on data batch
        """
        pass

    @abstractmethod
    def _predict_with_loss(self, batch, batchlabs):
        """
        Predict for given batch and return loss compared to labels in batchlabs

        Parameters
        ----------
        batch : image data
        batchlabs : corresponding label data

        Returns
        -------
        tuple of ndarray prediction and losses
        """
        pass

    @abstractmethod
    def _set_session(self, sess, cachefolder):
        pass

    @abstractmethod
    def _save(self, f):
        """
        Save to file f in current framework

        Parameters
        ----------
        f : location to save model at

        """
        pass

    @abstractmethod
    def _load(self, f):
        """
        Load model in current framework from f

        Parameters
        ----------
        f : location of stored model

        """
        pass

    @abstractmethod
    def get_globalstep(self):
        """
        Return number of iterations this model has been trained in

        Returns
        -------
        int : iteration count
        """
        pass

    def train(self):
        """
        Measures and logs time when performing data sampling and training iteration.
        """
        start_time = time.time()
        batch, batchlabs = self.trdc.random_sample(batch_size=self.batch_size)
        time_after_loading = time.time()
        loss = self._train(batch, batchlabs)
        self.currit += 1
        end_time = time.time()
        if (self.currit % self.print_each == 0):
            logging.getLogger("eval").info("it: {}, time: [i/o: {}, processing: {}, all: {}], loss: {}"
                                           .format(self.currit,
                                                   np.round(time_after_loading - start_time, 6),
                                                   np.round(end_time - time_after_loading, 6),
                                                   np.round(end_time - start_time, 6),
                                                   loss))
        return loss

    def test_scores(self, pred, ref):
        """
        Evaluates all selected scores between reference data ref and prediction pred.

        Parameters
        ----------
        pred : ndarray
            prediction, as probability distributions per pixel / voxel
        ref : ndarray
            labelmap, either as probability distributions per pixel / voxel or as label map

        """
        ref = np.int32(np.expand_dims(ref.squeeze(), 0))
        pred = np.expand_dims(pred.squeeze(), 0)
        if pred.shape != ref.shape:
            tar2 = np.zeros((np.prod(pred.shape[:-1]), pred.shape[-1]))
            tar2[np.arange(np.prod(pred.shape[:-1])), ref.flatten()] = 1
            ref = tar2.reshape(pred.shape)

        res = {}
        eps = 1e-8
        nclasses = self.model.nclasses
        if self.binary_evaluation:
            enc_ref = np.argmax(ref, -1)
            enc_pred = nclasses * np.argmax(pred, -1)
            enc_both = enc_ref + enc_pred
            bins = np.bincount(enc_both.flatten(), minlength=nclasses ** 2).reshape((nclasses, nclasses))
        if self.show_dice:
            res["dice"] = [bins[c, c] * 2 / (np.sum(bins, -1)[c] + np.sum(bins, -2)[c] + eps) for c in
                           range(nclasses)]
        if self.show_f05 or self.show_f2:
            precision = np.array([bins[c, c] / (np.sum(bins, -1)[c] + eps) for c in range(nclasses)])
            recall = np.array([bins[c, c] / (np.sum(bins, -2)[c] + eps) for c in range(nclasses)])
        if self.show_f05:
            beta2 = 0.5 ** 2
            res["f05"] = (1 + beta2) * precision * recall / ((beta2 * precision) + recall + eps)
        if self.show_f1:
            res["f1"] = [bins[c, c] * 2 / (np.sum(bins, -2)[c] + np.sum(bins, -1)[c] + eps) for c in
                         range(nclasses)]
        if self.show_f2:
            beta2 = 2 ** 2
            res["f2"] = (1 + beta2) * precision * recall / (beta2 * precision + recall + eps)
        if self.show_cross_entropy:
            res["cross_entropy"] = np.mean(np.sum(ref * np.log(pred + eps), -1))
        if self.show_l2:
            res["l2"] = np.mean(np.sum((ref - pred) ** 2, -1))
        return res

    def test_all_random(self, batch_size=None, dc=None, resample=True):
        """
        Test random samples

        Parameters
        ----------
        batch_size : int
            minibatch size to compute on
        dc : datacollection instance, optional
            datacollection to sample from
        resample : bool
            indicates if we need to sample before evaluating

        Returns
        -------
        tuple of loss and prediction ndarray
        """
        if dc is None:
            dc = self.valdc
        if batch_size is None:
            batch_size = self.batch_size
        if self.validate_same:
            dc.randomstate.seed(12345677)
        if resample:
            self.testbatch, self.testbatchlabs = dc.random_sample(batch_size=batch_size)
        loss, prediction = self._predict_with_loss(self.testbatch, self.testbatchlabs)

        return loss, prediction

    def test_all_available(self, batch_size=None, dc=None, return_results=False, dropout=None, testing=False):
        """
        Completely evaluates each full image in tps using grid sampling.

        Parameters
        ----------
        batch_size : int
            minibatch size to compute on
        dc : datacollection instance, optional
            datacollection to sample from
        return_results : bool
            should results be returned or stored right away?
        dropout : float
            keeprate of dropconnect for inference
        testing

        Returns
        -------
        either tuple of predictions and errors or only errors, depending on return_results flag
        """
        if dc is None:
            dc = self.tedc

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
            res = np.zeros(list(shape) + [self.model.nclasses], dtype=np.float32)
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
                    slicesa = tuple(slicesa)
                    slicesb = tuple(slicesb)
                    certainty[slicesa] *= np.arange(1.0 / (pp + 1), 1, 1.0 / (pp + 1)).reshape(reshapearr)
                    certainty[slicesb] *= np.arange(1.0 / (pp + 1), 1, 1.0 / (pp + 1))[::-1].reshape(reshapearr)

            certainty = certainty.reshape([1] + w + [1])
            # read, compute, merge, write back
            for subvol, _, imin, imax in volgen:
                if self.evaluate_uncertainty_times > 1:
                    preds = []
                    for i in range(self.evaluate_uncertainty_times):
                        preds.append(self._predict(subvol, dropout, testing))
                        logging.getLogger('eval').debug(
                            'evaluated run {} of subvolume from {} to {}'.format(i, imin, imax))
                    pred = np.mean(np.asarray(preds), 0)
                    uncert = np.std(np.asarray(preds), 0)
                    preds = [x * certainty for x in preds]
                else:
                    pred = self._predict(subvol, dropout, testing)
                    logging.getLogger('eval').debug('evaluated subvolume from {} to {}'.format(imin, imax))

                pred *= certainty
                # now reembed it into array
                wrongmin = [int(abs(x)) if x < 0 else 0 for x in imin]
                wrongmax = [int(x) if x < 0 else None for x in (shape - imax)]
                mimin = np.asarray(np.maximum(0, imin), dtype=np.int32)
                mimax = np.asarray(np.minimum(shape, imax), dtype=np.int32)
                slicesaa = [slice(mimina, miminb) for mimina, miminb in zip(mimin, mimax)]
                slicesaa.append(slice(None))
                slicesaa = tuple(slicesaa)
                slicesbb = [0]
                slicesbb.extend(slice(wrongmina, wrongminb) for wrongmina, wrongminb in zip(wrongmin, wrongmax))
                slicesbb.append(slice(None))
                slicesbb = tuple(slicesbb)

                res[slicesaa] += pred[slicesbb]
                if self.evaluate_uncertainty_times > 1:
                    uncert *= certainty
                    uncertres[slicesaa] += \
                        uncert[slicesbb]
                    if self.evaluate_uncertainty_saveall:
                        for j in range(self.evaluate_uncertainty_times):
                            allres[j, slicesaa] += preds[i][slicesbb]

            # normalize again:
            if self.evaluate_uncertainty_times > 1 and not return_results:
                uncertres /= np.sum(res, -1).reshape(list(res.shape[:-1]) + [1])
                dc.save(uncertres, os.path.join(file, "std-" + self.estimatefilename), tporigin=file)
                if self.evaluate_uncertainty_saveall:
                    for j in range(self.evaluate_uncertainty_times):
                        dc.save(allres[j], os.path.join(file, "iter{}-".format(j) + self.estimatefilename),
                                tporigin=file)
            if np.min(p) < 0:
                res[np.where(np.sum(res, -1) < 1e-8)] = [1] + [0 for _ in range(self.model.nclasses - 1)]
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
                logging.getLogger('eval').warning(
                    'was not able to save test scores, is ground truth available?')
                logging.getLogger('eval').warning('{}'.format(e))
            if return_results:
                full_vols.append([name, file, res])
            else:
                if not self.only_save_labels:
                    dc.save(res, os.path.join(file, self.estimatefilename + "-probdist"), tporigin=file)
                dc.save(np.uint8(np.argmax(res, -1)), os.path.join(file, self.estimatefilename + "-labels"),
                        tporigin=file)
            logging.getLogger('eval').info('evaluation took {} seconds'.format(time.time() - lasttime))
            lasttime = time.time()
        if return_results:
            return full_vols, errs
        else:
            return errs

    def load(self, f):
        """
        loads model at location f from disk

        Parameters
        ----------
        f : str
            location of stored model
        """
        self._load(f)

        states = {}
        try:
            pickle_name = f.rsplit('-', 1)[0] + ".pickle"
            states = pickle.load(open(pickle_name, "rb"))
        except Exception as e:
            logging.getLogger('eval').warning('there was no randomstate pickle named {} around'.format(pickle_name))
        if "trdc" in states:
            self.trdc.set_states(states['trdc'])
        else:
            self.trdc.set_states(None)
        if "tedc" in states:
            self.tedc.set_states(states['tedc'])
        else:
            self.tedc.set_states(None)
        if "valdc" in states:
            self.valdc.set_states(states['valdc'])
        else:
            self.valdc.set_states(None)
        if 'epoch' in states:
            self.current_epoch = states['epoch']
        else:
            self.current_epoch = 0
        if 'iteration' in states:
            self.current_iteration = states['iteration']
        else:
            self.current_iteration = 0
        self.current_iteration = self.current_iteration

    def save(self, f):
        """
        saves model to disk at location f

        Parameters
        ----------
        f : str
            location to save model to
        """
        ckpt = self._save(f)
        trdc = self.trdc.get_states()
        tedc = self.tedc.get_states()
        valdc = self.valdc.get_states()
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
        return ckpt

    def add_summary_simple_value(self, text, value):
        raise NotImplementedError("this needs to be implemented and only works with tensorflow backend.")

    def set_session(self, sess, cachefolder, train=False):
        return None

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

generate_defaults_info(SupervisedEvaluation)

