__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import inspect
import itertools
import logging
import os
import re
import shutil
import signal
import sys
import time as time
import datetime
import csv
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from helper import argget, force_symlink, notify_user

try:
    import _pickle as pickle  # cPickle is now _pickle (TODO: CHECK)
except Exception:
    import pickle
import copy

ignore_signal = False


class Runner(object):
    ignore_signal = True

    def __init__(self, evaluationinstance, **kw):
        self.origargs = copy.deepcopy(kw)
        # prelogging:
        experiments = argget(kw, 'experimentloc', os.path.expanduser('~/experiments'))
        self.runfile = [f[1] for f in inspect.stack() if re.search("RUN.*\.py", f[1])][0]

        self.experiments_postfix = argget(kw, 'experiments_postfix', "")
        experiments_nots = os.path.join(experiments,
                                        '{}'.format(
                                            self.runfile[self.runfile.index("RUN_") + 4:-3] + self.experiments_postfix))
        self.experiments = os.path.join(experiments_nots, str(int(time.time())))
        self.cachefolder = os.path.join(self.experiments, 'cache')
        os.makedirs(self.cachefolder)
        # logging:
        loggers = [logging.getLogger(n) for n in ['model', 'eval', 'runner', 'helper', 'data']]
        formatter = logging.Formatter('%(asctime)s %(name)s\t%(levelname)s:\t%(message)s')
        logfile = argget(kw, 'logfile', os.path.join(self.cachefolder, 'log.txt'))
        fh = logging.FileHandler(logfile)
        fh.setLevel(argget(kw, 'logfileloglvl', logging.DEBUG))
        fh.setFormatter(formatter)

        self.only_cpu = argget(kw, 'only_cpu', False)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch.setLevel(argget(kw, 'loglvl', logging.WARNING))
        for logger in loggers:
            logger.addHandler(ch)
            logger.setLevel(logging.DEBUG)
            logger.addHandler(fh)

        for k in self.origargs:
            logging.getLogger('runner').info('arg {}:{}'.format(k, self.origargs[k]))
        self.ev = evaluationinstance
        for k in self.ev.origargs:
            logging.getLogger('eval').info('arg {}:{}'.format(k, self.ev.origargs[k]))
        for k in self.ev.trdc.origargs:
            logging.getLogger('data').info(' trdc arg {}:{}'.format(k, self.ev.trdc.origargs[k]))
        for k in self.ev.tedc.origargs:
            logging.getLogger('data').info(' tedc arg {}:{}'.format(k, self.ev.tedc.origargs[k]))
        for k in self.ev.valdc.origargs:
            logging.getLogger('data').info('valdc arg {}:{}'.format(k, self.ev.valdc.origargs[k]))
        for k in self.ev.model.origargs:
            logging.getLogger('model').info('arg {}:{}'.format(k, self.ev.model.origargs[k]))

        self.episodes = argget(kw, 'episodes', ['train', 'evaluate'])
        self.epochs = argget(kw, 'epochs', 1)
        self.batch_size = argget(kw, 'batch_size', 1)
        self.its_per_epoch = argget(kw, 'its_per_epoch', self.ev.trdc.get_data_dims()[0] // self.batch_size)
        self.checkpointfile = argget(kw, 'checkpointfile', None)
        self.plotfolder = os.path.join(self.experiments, 'plot')
        self.plot_scaling = argget(kw, 'plot_scaling', 1e-8)
        self.display_each = argget(kw, 'display_each', 100)
        self.test_each = argget(kw, 'test_each', self.display_each)
        self.save_each = argget(kw, 'save_each', self.display_each)
        self.plot_each = argget(kw, 'plot_each', self.display_each)
        self.test_size = argget(kw, 'test_size', 1)  # batch_size for tests
        self.test_iters = argget(kw, 'test_iters', 1)
        self._test_pick_iteration = argget(kw, 'test_first', ifset=0, default=self.test_each - 1)
        self.perform_n_times_full_validation = argget(kw, 'perform_n_times_full_validation', 0)
        self.perform_n_times_full_validation_dropout = argget(kw, 'perform_n_times_full_validation_dropout', 0.5)
        self.show_testing_results = argget(kw, 'show_testing_results', False)
        force_symlink(self.experiments, os.path.join(experiments_nots, "latest"))
        os.makedirs(self.plotfolder)
        self.printIt = argget(kw, "print_testing_results", True)
        self.gpubound = argget(kw, 'gpubound', 1)
        self.notifyme = argget(kw, 'notifyme', None)
        self.results_to_csv = argget(kw, 'results_to_csv', False)

        self.train_losses = []
        self.test_losses = []
        self.val_losses = []

        # remove full parameters since it would not be an ignored parameter and confuse us
        kw.pop('fullparameters')
        if kw:
            logging.getLogger('runner').warning('the following args were ignored: {}'.format(kw))

    def validation(self, showIt=True, name=time.time()):

        for i in range(self.test_iters):
            allres = []
            testbatches = []
            testlabs = []
            errors = []
            if self.perform_n_times_full_validation:
                res, error = self.ev.test_all_available(batch_size=self.test_size, dc=self.ev.valdc,
                                                        return_results=True,
                                                        dropout=self.perform_n_times_full_validation_dropout)
                allres.extend(res)
                errors.append(error)
            else:
                error, result = self.ev.test_all_random(batch_size=self.test_size, dc=self.ev.valdc)
                currtp = self.ev.valdc.tps[self.ev.valdc.curr_tps]
                filename = os.path.split(currtp)
                filename = filename[-1] if len(filename[-1]) else os.path.basename(filename[0])
                res = [filename, currtp, result]
                allres.append(res)
                if error:
                    if np.isscalar(error):
                        error = {'ce': error}
                    errors.append(error)
                testbatches.append(self.ev.testbatch)
                testlabs.append(self.ev.testbatchlabs)

            try:
                for n, ee in error:
                    logging.getLogger('runner').info("{}: {}".format(n, ee))
            except:
                logging.getLogger('runner').info(error)


            def save_all_res(cachefolder, rr, dc, tname):
                for rit, r in enumerate(rr):
                    if showIt:
                        dc.save(r[2], os.path.join(cachefolder, '{}-{}-{}-pred'.format(tname, r[0], rit)), tporigin=r[1])
                        dc.save(np.int8(np.argmax(r[2], axis=-1)),
                                os.path.join(cachefolder, "{}-{}-{}-am".format(tname, r[0], rit)), tporigin=r[1])
                        logging.getLogger('runner').info("saved validation test files in cache")

            Thread(target=save_all_res, args=(self.cachefolder, res, self.ev.valdc, name,)).start()

        errors = list(itertools.chain.from_iterable(errors))

        minerrors, avgerrors, medianerrors, maxerrors = self.calc_min_mean_median_max_errors(errors)

        logging.getLogger('runner').info("min    errors {}".format(minerrors))
        logging.getLogger('runner').info("mean   errors {}".format(avgerrors))
        logging.getLogger('runner').info("median errors {}".format(medianerrors))
        logging.getLogger('runner').info("max    errors {}".format(maxerrors))

        if self.results_to_csv:
            self.write_error_to_csv(errors, 'validation_scores.csv', minerrors, avgerrors, medianerrors, maxerrors)

        return avgerrors

    def _finish(self, signal):
        if signal:
            fname = 'interrupt'
        else:
            fname = 'final'

        if self.notifyme is not None:
            if signal != 0:
                notify_user(self.notifyme['chat_id'], self.notifyme['token'], message='Process was killed')
        self.save(fname + '.ckpt')
        with open(os.path.join(self.cachefolder, 'trainloss.pickle'), 'wb') as f:
            pickle.dump(self.train_losses, f)
        with open(os.path.join(self.cachefolder, 'valloss.pickle'), 'wb') as f:
            pickle.dump(self.val_losses, f)
        with open(os.path.join(self.cachefolder, 'testloss.pickle'), 'wb') as f:
            pickle.dump(self.test_losses, f)

    def train(self):

        Runner.ignore_signal = False  # lets the signal handler work only once

        def signal_handler(signal, frame):
            logging.getLogger('runner').warning('You pressed Ctrl+C! shutting down')
            try:
                if Runner.ignore_signal:
                    return
            except:
                pass
            ignore_signal = True
            self._finish(1)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        logging.getLogger('runner').info('If you press Ctrl+C, we save the model before exit')
        # train
        epoch = -1
        starttime = time.time()
        epochs = self.epochs if self.epochs > 0 else 1
        for epoch in range(self.ev.current_epoch, epochs):
            self.ev.current_epoch = epoch
            logging.getLogger('runner').info(
                "epoch {}/{}: running for {} batches of {}:".format(epoch, self.epochs, self.its_per_epoch,
                                                                    self.batch_size))
            for it in range(self.ev.current_iteration, self.its_per_epoch):
                self.ev.current_iteration = it
                a = time.time()
                loss = self.ev.train(batch_size=self.batch_size)
                #logging.getLogger('runner').info(
                #    "with img loading: {} {}".format(time.time() - a, np.asarray(loss).flatten()))
                self.train_losses.append([epoch, it, loss])

                if it % self.test_each == self._test_pick_iteration:
                    error = self.validation(showIt=self.show_testing_results, name=it)
                    self.val_losses.append([epoch, it, [e + self.plot_scaling for k, e in error.items()]])

                    difftime = time.time() - starttime
                    divisor = (it + 1) + epoch * self.its_per_epoch
                    logging.getLogger('runner').info(
                        "{}/{} of {}/{} in {}/{}| {} left".format(it, self.its_per_epoch, epoch, self.epochs,
                                                                  difftime,
                                                                  difftime / divisor * self.its_per_epoch * self.epochs,
                                                                  difftime / divisor * self.its_per_epoch * self.epochs - difftime))
                if it % self.save_each == self.save_each - 1:
                    self.save('temp-{}.ckpt'.format(epoch))

            self.ev.current_iteration = 0

        self.ev.current_epoch = self.epochs
        self._finish(0)

    def save(self, filename):
        globalstep = self.ev.sess.run(self.ev.model.global_step)
        abspath = os.path.join(self.cachefolder, filename)
        self.ev.save(abspath)
        self.checkpointfile = abspath+'-{}'.format(globalstep)
        logging.getLogger('runner').info('Saved checkpoint {}'.format(filename+'-{}'.format(globalstep)))

    def write_error_to_csv(self, errors, filename, minerrors, avgerrors, medianerrors, maxerrors):
        try:

            with open(os.path.join(self.cachefolder, filename), 'a') as csvfile:

                globalstep = self.ev.sess.run(self.ev.model.global_step)
                currenttime = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
                ckptfile = self.checkpointfile # if self.checkpointfile is a list -> adapt ckptfile

                evaluationWriter = csv.writer(csvfile)
                evaluationWriter.writerow(['score', 'label'] + [errors[i][0] for i in range(0, len(errors))] + ['checkpoint', 'iteration', 'time-stamp','score','label','min','mean','median','max'])
                for key in errors[0][1].keys():
                    try:
                        for label in range(0, len(errors[0][1][key])):
                            evaluationWriter.writerow([key]
                                                      + ['label' + str(label)]
                                                      + [str(errors[i][1][key][label]) for i in range(0, len(errors))]
                                                      + [ckptfile, str(globalstep), currenttime, key]
                                                      + ['label' + str(label)]
                                                      + [str(minerrors[key][label]), str(avgerrors[key][label]),
                                                         str(medianerrors[key][label]), str(maxerrors[key][label])])
                    except:
                        evaluationWriter.writerow([key]
                                                  + ['all_labels']
                                                  + [str(errors[i][1][key]) for i in range(0, len(errors))]
                                                  + [ckptfile, str(globalstep), currenttime, key, 'all_labels']
                                                  + [str(minerrors[key]), str(avgerrors[key]),
                                                     str(medianerrors[key]), str(maxerrors[key])])
        except:
            logging.getLogger('runner').warning('could not write error to ' + filename)


    def calc_min_mean_median_max_errors(self, errors):
        avgerrors = {}
        minerrors = {}
        medianerrors = {}
        maxerrors = {}
        for k in errors[0][1].keys():
            val = [errors[i][1][k] for i in range(len(errors)) if k in errors[i][1].keys()]
            avgerrors[k] = np.mean(val, 0)
            minerrors[k] = np.nanmin(val, 0)
            medianerrors[k] = np.median(val, 0)
            maxerrors[k] = np.nanmax(val, 0)

        if self.ev.use_tensorboard:
            for k, v in avgerrors.items():
                if np.isscalar(v):
                    v = [v]
                for c, vv in enumerate(v):
                    try:
                        summary = tf.Summary()
                        summary.value.add(tag='validation-mean-{}-{}'.format(k, c), simple_value=vv)
                        self.ev.train_writer.add_summary(summary)
                    except Exception as e:
                        logging.getLogger('runner').warning('could not save {} as scalar value'.format(vv))

        return minerrors, avgerrors, medianerrors, maxerrors


    def test(self):
        self.ev.tedc.p = np.int32(self.ev.tedc.p)
        errors = self.ev.test_all_available(batch_size=1, testing=True)
        if self.results_to_csv and len(errors):
            minerrors, avgerrors, medianerrors, maxerrors = self.calc_min_mean_median_max_errors(errors)
            self.write_error_to_csv(errors, 'testing_scores.csv', minerrors, avgerrors, medianerrors, maxerrors)



    def run(self, **kw):
        # save this file as txt to cachefolder:
        if self.only_cpu:
            config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            if self.gpubound < 1:
                config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=self.gpubound))
            else:
                config = tf.ConfigProto()

        shutil.copyfile(self.runfile, os.path.join(self.cachefolder, 'runfile.py'))

        if "train" in self.episodes:
            with tf.Session(config=config) as sess:
                self.ev.set_session(sess, self.cachefolder)
                if self.checkpointfile:
                    self.ev.load(self.checkpointfile)
                self.train()

        if "test" in self.episodes or "evaluate" in self.episodes:
            self.use_tensorboard = False # no need, since we evaluate everything anyways.
            with tf.Session(config=config, graph=self.ev.test_graph) as sess:
                self.ev.set_session(sess, self.cachefolder)
                if self.checkpointfile:
                    self.ev.load(self.checkpointfile)
                self.test()
        if self.notifyme:
            notify_user(self.notifyme['chat_id'], self.notifyme['token'], message='{} has/have finished'.format(" and ".join(self.episodes)))
