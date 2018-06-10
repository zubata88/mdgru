__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import copy
import logging
import os
import pickle
import time
from abc import abstractmethod

import numpy as np

from helper import argget


class SupervisedEvaluation(object):
    def __init__(self, model, collectioninst, kw):
        self.origargs = copy.deepcopy(kw)
        self.dropout_rate = argget(kw, "dropout_rate", 0.5)
        self.current_epoch = 0
        self.current_iteration = 0
        self.trdc = collectioninst["train"]
        self.tedc = collectioninst["test"]
        self.valdc = collectioninst["validation"]
        self.nclasses = argget(kw, "nclasses", 2, keep=True)
        self.currit = 0
        self.namespace = argget(kw, "namespace", "default")
        self.only_save_labels = argget(kw, "only_save_labels", False)
        self.batch_size = argget(kw, "batch_size", 1)
        self.validate_same = argget(kw, "validate_same", False)
        self.evaluate_uncertainty_times = argget(kw, "evaluate_uncertainty_times", 1)
        self.evaluate_uncertainty_dropout = argget(kw, "evaluate_uncertainty_dropout",
                                                   1.0)  # these standard values ensure that we dont evaluate uncertainty if nothing was provided.
        self.evaluate_uncertainty_saveall = argget(kw, "evaluate_uncertainty_saveall", False)

        self.f05 = argget(kw, "show_f05", True)
        self.f1 = argget(kw, "show_f1", True)
        self.f2 = argget(kw, "show_f2", True)
        self.l2 = argget(kw, "show_l2_loss", True)
        self.dice = argget(kw, "show_dice", not self.f1)
        self.cross_entropy = argget(kw, "show_cross_entropy_loss", True)
        self.binary_evaluation = self.dice or self.f1 or self.f05 or self.f2
        self.estimatefilename = argget(kw, "estimatefilename", "estimate")
        self.gpu = argget(kw, "gpu", 0)

    @abstractmethod
    def _train(self):
        """ Performs one training iteration in respective framework and returns loss(es)"""
        raise Exception("This needs to be implemented depending on the framework")

    @abstractmethod
    def _predict(self, batch, dropout, testing):
        pass

    @abstractmethod
    def _predict_with_loss(self, batch, batchlabs):
        pass

    @abstractmethod
    def _set_session(self, sess, cachefolder):
        pass

    @abstractmethod
    def _save(self, f):
        pass

    @abstractmethod
    def _load(self, f):
        pass

    def train(self):
        """ Measures and logs time for data sampling and training iteration."""
        start_time = time.time()
        batch, batchlabs = self.trdc.random_sample(batch_size=self.batch_size)
        time_after_loading = time.time()
        loss = self._train(batch, batchlabs)
        self.currit += 1
        end_time = time.time()
        logging.getLogger("eval").info("it: {}, time: [i/o: {}, processing: {}, all: {}], loss: {}"
                                       .format(self.currit,
                                               np.round(time_after_loading - start_time, 6),
                                               np.round(end_time - time_after_loading, 6),
                                               np.round(end_time - start_time, 6),
                                               loss))
        return loss

    def test_scores(self, pred, ref):
        """ Evaluates all activated scores between ref and pred."""
        ref = np.int32(np.expand_dims(ref.squeeze(), 0))
        pred = np.expand_dims(pred.squeeze(), 0)
        if pred.shape != ref.shape:
            tar2 = np.zeros((np.prod(pred.shape[:-1]), pred.shape[-1]))
            tar2[np.arange(np.prod(pred.shape[:-1])), ref.flatten()] = 1
            ref = tar2.reshape(pred.shape)

        res = {}
        eps = 1e-8
        if self.binary_evaluation:
            enc_ref = np.argmax(ref, -1)
            enc_pred = self.nclasses * np.argmax(pred, -1)
            enc_both = enc_ref + enc_pred
            bins = np.bincount(enc_both.flatten(), minlength=self.nclasses ** 2).reshape((self.nclasses, self.nclasses))
        if self.dice:
            res["dice"] = [bins[c, c] * 2 / (np.sum(bins, -1)[c] + np.sum(bins, -2)[c] + eps) for c in
                           range(self.nclasses)]
        if self.f05 or self.f2:
            precision = np.array([bins[c, c] / (np.sum(bins, -1)[c] + eps) for c in range(self.nclasses)])
            recall = np.array([bins[c, c] / (np.sum(bins, -2)[c] + eps) for c in range(self.nclasses)])
        if self.f05:
            beta2 = 0.5 ** 2
            res["f05"] = (1 + beta2) * precision * recall / ((beta2 * precision) + recall + eps)
        if self.f1:
            res["f1"] = [bins[c, c] * 2 / (np.sum(bins, -2)[c] + np.sum(bins, -1)[c] + eps) for c in
                         range(self.nclasses)]
        if self.f2:
            beta2 = 2 ** 2
            res["f2"] = (1 + beta2) * precision * recall / (beta2 * precision + recall + eps)
        if self.cross_entropy:
            res["cross_entropy"] = np.mean(np.sum(ref * np.log(pred + eps), -1))
        if self.l2:
            res["l2"] = np.mean(np.sum((ref - pred) ** 2, -1))
        return res

    def test_all_random(self, batch_size=None, dc=None, resample=True):
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
                slicesbb = [0]
                slicesbb.extend(slice(wrongmina, wrongminb) for wrongmina, wrongminb in zip(wrongmin, wrongmax))
                slicesbb.append(slice(None))

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
                    'was not able to save test scores, even though ground truth was available.')
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
        '''loads model at location f from disk'''
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
        '''saves model to disk at location f'''
        self._save(f)
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

    def add_summary_simple_value(self, text, value):
        raise NotImplementedError("this needs to be implemented and only works with tensorflow backend.")

    def get_train_session(self, cachefolder):
        return self

    def get_test_session(self, cachefolder):
        return self

    def set_session(self, sess, cachefolder, train=False):
        return None

    def __enter__(self):
        pass

    def __exit__(self):
        pass
