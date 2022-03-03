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

import numpy as np

from mdgru.helper import argget, force_symlink, notify_user, compile_arguments, generate_defaults_info

try:
    import _pickle as pickle  # cPickle is now _pickle (TODO: CHECK)
except Exception:
    import pickle
import copy
import hashlib
import json

ignore_signal = False


class Runner(object):
    ignore_signal = True
    _defaults = {
        'test_each': {'value': 2500, 'type': int, 'help': 'validate after # training iterations', 'name': 'validate_each'},
        'save_each': {'value': None, 'type': int, 'help': 'save after # training iterations'},
        'plot_each': {'value': 2500, 'type': int},  # , 'help': 'plot each # training iterations'},
        'test_size': {'value': 1},
        'test_iters': {'value': 1, 'type': int,
                       'help': 'number of validations to perform on random samples. Only makes sense if full_image_validation is not set',
                       'name': 'validation_iterations'},
        'test_first': {'value': False, 'help':'Perform validation on the untrained model as well.', 'name': 'validate_first'},
        'perform_full_image_validation': {'value': True, 'invert_meaning': 'dont_', 'help': 'Use random samples instead of the complete validation images'},
        'save_validation_results': {'value': True, 'invert_meaning': 'dont_', 'help': 'Do not save validation results on the disk' },
        'notifyme': {'value': None, 'nargs':'?', 'type':str,
                                  'help':'Experimental feature that when something goes amiss, '
                                       'this telegram chat id will be used to inform the '
                                       'respective user of the error. This requires a file called config.json'
                                       ' in the same folder as this file, containing a simple dict structure as'
                                       ' follows: {"chat_id": CHATID, "token": TOKEN}, where CHATID and TOKEN '
                                       'have to be created with Telegrams BotFather. The chatid from config can be '
                                       'overriden using a parameter together with this option.'},
        'results_to_csv': {'value': True, 'help': 'Do not create csv with validation results'},
        'checkpointfiles': {'value': None, 'help': 'provide checkpointfile for this template. If no modelname is provided, '
                                       'we will infer one from this file. Multiple files are only allowed if '
                                       'only_test is set. If the same number of optionnames are provided, they will '
                                       'be used to name the different results. If only one optionname is provided, they '
                                       'will be numbered in order and the checkpoint filename will be included in the '
                                       'result file.', 'nargs':'+', 'name':'ckpt'},
        'epochs': {'value': 1, 'help':'Number of times through the training dataset. Cant be used together with "iterations"'},
        'iterations': {'value': None, 'type': int, 'help': 'Number of iterations to perform. Can only be set and makes sense if epochs is 1'},
        'only_test': {'value': False, 'help': 'Only perform testing. Requires at least one ckpt'},
        'only_train': {'value': False, 'help': 'Only perform training and validation.'},
        'experimentloc': os.path.expanduser('~/experiments'),
        'optionname': {'value': None, 'nargs':'+', 'help':'name for chosen set of options, if multiple checkpoints provided, there needs to be 1 or the same number of names here'},
        'fullparameters': None,
    }

    def __init__(self, evaluationinstance, **kw):
        """

        Parameters
        ----------
        evaluationinstance : instance of an evaluation class
            Will be used to call train and test routines on.
        """
        self.origargs = copy.deepcopy(kw)
        runner_kw, kw = compile_arguments(Runner, kw, transitive=False)
        for k, v in runner_kw.items():
            setattr(self, k, v)

        if self.save_each is None:
            self.save_each = self.test_each

        if self.notifyme:
            try:
                # import json
                data = json.load(open('../config.json'))
                nm = dict(chat_id=data['chat_id'], token=data['token'])
                try:
                    nm['chat_id'] = int(self.notifyme)
                except Exception:
                    pass
                finally:
                    self.notifyme = nm
            except:
                # we give up
                print('notifyme id not understood')

        # prelogging:
        # experiments = argget(kw, 'experimentloc', os.path.expanduser('~/experiments'))
        self.runfile = [f[1] for f in inspect.stack() if re.search("RUN.*\.py", f[1])][0]

        if self.optionname is None:
            self.optionname = [hashlib.sha256(json.dumps(self.fullparameters).encode('utf8')).hexdigest()]
        elif not isinstance(self.optionname, list):
            self.optionname = [self.optionname]

        self.estimatefilenames = self.optionname
        if isinstance(self.optionname, list):
            pf = "-".join(self.optionname)
            if len(pf) > 40:
                pf = pf[:39] + "..."
        else:
            pf = self.optionname
        self.experiments_postfix = '_' + pf
        experiments_nots = os.path.join(self.experimentloc,
                                        '{}'.format(
                                            self.runfile[self.runfile.index("RUN_") + 4:-3] + self.experiments_postfix))
        self.experiments = os.path.join(experiments_nots, str(int(time.time())))
        self.cachefolder = os.path.join(self.experiments, 'cache')
        os.makedirs(self.cachefolder)
        
        # Add Logging.FileHandler (StreamHandler was added already in RUN_mdgru.py)
        loggers = [logging.getLogger(n) for n in ['model', 'eval', 'runner', 'helper', 'data']]
        formatter = logging.Formatter('%(asctime)s %(name)s\t%(levelname)s:\t%(message)s')
        logfile = argget(kw, 'logfile', os.path.join(self.cachefolder, 'log.txt'))
        fh = logging.FileHandler(logfile)
        fh.setLevel(argget(kw, 'logfileloglvl', logging.DEBUG))
        fh.setFormatter(formatter)
        # ch = logging.StreamHandler()
        # ch.setFormatter(formatter)
        # ch.setLevel(argget(kw, 'loglvl', logging.DEBUG))        
        for logger in loggers:
            logger.setLevel(logging.DEBUG)
            # logger.addHandler(ch)
            logger.addHandler(fh)

        for k in self.origargs:
            logging.getLogger('runner').info('args runner {}:{}'.format(k, self.origargs[k]))
        self.ev = evaluationinstance
        for k in self.ev.origargs:
            logging.getLogger('runner').info('args eval/data/model {}:{}'.format(k, self.ev.origargs[k]))
        # for k in self.ev.trdc.origargs:
        #     logging.getLogger('data').info(' trdc arg {}:{}'.format(k, self.ev.trdc.origargs[k]))
        # for k in self.ev.tedc.origargs:
        #     logging.getLogger('data').info(' tedc arg {}:{}'.format(k, self.ev.tedc.origargs[k]))
        # for k in self.ev.valdc.origargs:
        #     logging.getLogger('data').info('valdc arg {}:{}'.format(k, self.ev.valdc.origargs[k]))
        # for k in self.ev.model.origargs:
        #     logging.getLogger('model').info('arg {}:{}'.format(k, self.ev.model.origargs[k]))
        if self.only_train or (self.ev.trdc == self.ev.tedc and self.ev.valdc != self.ev.trdc):
            self.episodes = ['train']
        elif self.only_test or (self.ev.trdc == self.ev.tedc and self.ev.valdc == self.ev.tedc):
            self.episodes = ['evaluate']
        else:
            self.episodes = ['train', 'evaluate']
        # self.episodes = argget(kw, 'episodes', ['train', 'evaluate'])
        # self.epochs = argget(kw, 'epochs', 1)
        if self.iterations is None:
            self.its_per_epoch = self.ev.trdc.get_data_dims()[0] // self.ev.batch_size
        else:
            self.epochs = 0
            self.its_per_epoch = self.iterations
        # self.its_per_epoch = argget(kw, 'its_per_epoch', self.ev.trdc.get_data_dims()[0] // self.ev.batch_size)
        # self.checkpointfiles = argget(kw, 'checkpointfiles', None)
        self.estimatefilenames = self.optionname#argget(kw, 'estimatefilenames', None)
        if isinstance(self.checkpointfiles, list):
            if 'train' in self.episodes and len(self.checkpointfiles) > 1:
                logging.getLogger('runner').error('Multiple checkpoints are only allowed if only testing is performed.')
                exit(1)
        else:
            self.checkpointfiles = [self.checkpointfiles]
        # if not isinstance(self.estimatefilenames, list):
        #     self.estimatefilenames = [self.estimatefilenames]
        if len(self.checkpointfiles) != len(self.estimatefilenames):
            if len(self.estimatefilenames) != 1:
                logging.getLogger('runner').error('Optionnames must match number of checkpoint files or have length 1!')
                exit(1)
            else:
                self.estimatefilenames = [self.estimatefilenames[0] + "-{}-{}".format(i, os.path.basename(c))
                                          for i, c in enumerate(self.checkpointfiles)]

        self.plotfolder = os.path.join(self.experiments, 'plot')
        self.plot_scaling = argget(kw, 'plot_scaling', 1e-8)


        # self.display_each = argget(kw, 'display_each', 100)
        # self.test_each = argget(kw, 'test_each', 100)
        # self.save_each = argget(kw, 'save_each', self.test_each)
        # self.plot_each = argget(kw, 'plot_each', self.test_each)
        # self.test_size = argget(kw, 'test_size', 1)  # batch_size for tests
        # self.test_iters = argget(kw, 'test_iters', 1)
        self._test_pick_iteration = self.test_each-1 if not self.test_first else 0
        # self._test_pick_iteration = argget(kw, 'test_first', ifset=0, default=self.test_each - 1)
        # self.perform_full_image_validation = argget(kw, 'perform_full_image_validation', 1)
        # self.save_validation_results = argget(kw, 'show_testing_results', False)
        force_symlink(self.experiments, os.path.join(experiments_nots, "latest"))
        os.makedirs(self.plotfolder)
        # self.notifyme = argget(kw, 'notifyme', False)
        # self.results_to_csv = argget(kw, 'results_to_csv', False)

        self.train_losses = []
        self.test_losses = []
        self.val_losses = []

        # remove full parameters since it would not be an ignored parameter and confuse us
        # kw.pop('fullparameters')
        if kw:
            logging.getLogger('runner').warning('the following args were ignored: {}'.format(kw))

    def validation(self, showIt=True, name=time.time()):

        for i in range(self.test_iters):
            allres = []
            testbatches = []
            testlabs = []
            errors = []
            if self.perform_full_image_validation:
                res, error = self.ev.test_all_available(batch_size=self.test_size, dc=self.ev.valdc,
                                                        return_results=True)
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
                        dc.save(r[2], os.path.join(cachefolder, '{}-{}-{}-pred'.format(tname, r[0], rit)),
                                tporigin=r[1])
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
        self.save(fname)
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
                                                                    self.ev.batch_size))
            for it in range(self.ev.current_iteration, self.its_per_epoch):
                self.ev.current_iteration = it
                a = time.time()
                loss = self.ev.train()
                # logging.getLogger('runner').info(
                #    "with img loading: {} {}".format(time.time() - a, np.asarray(loss).flatten()))
                self.train_losses.append([epoch, it, loss])

                if it % self.test_each == self.test_each - 1 or (self.test_first and it == 0):
                    error = self.validation(showIt=self.save_validation_results, name=it)
                    self.val_losses.append([epoch, it, [e + self.plot_scaling for k, e in error.items()]])

                    difftime = time.time() - starttime
                    divisor = (it + 1) + epoch * self.its_per_epoch
                    logging.getLogger('runner').info(
                        "{}/{} of {}/{} in {}/{}| {} left".format(it, self.its_per_epoch, epoch, self.epochs,
                                                                  difftime,
                                                                  difftime / divisor * self.its_per_epoch * self.epochs,
                                                                  difftime / divisor * self.its_per_epoch * self.epochs - difftime))
                if it % self.save_each == self.save_each - 1:
                    self.save('temp-epoch{}'.format(epoch) if self.epochs > 0 else 'temp')

            self.ev.current_iteration = 0

        self.ev.current_epoch = self.epochs
        self._finish(0)

    def save(self, filename):
        abspath = os.path.join(self.cachefolder, filename)
        self.checkpointfiles[0] = self.ev.save(abspath)
        logging.getLogger('runner').info('Saved checkpoint {}'.format(self.checkpointfiles[0]))

    def write_error_to_csv(self, errors, filename, minerrors, avgerrors, medianerrors, maxerrors):
        try:

            with open(os.path.join(self.cachefolder, filename), 'a') as csvfile:

                globalstep = self.ev.get_globalstep()
                currenttime = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
                ckptfile = self.checkpointfiles[0]  # if self.checkpointfile is a list -> adapt ckptfile

                evaluationWriter = csv.writer(csvfile)
                evaluationWriter.writerow(
                    ['score', 'label'] + [errors[i][0] for i in range(0, len(errors))] + ['checkpoint', 'iteration',
                                                                                          'time-stamp', 'score',
                                                                                          'label', 'min', 'mean',
                                                                                          'median', 'max'])
                for key in sorted(errors[0][1].keys()):
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
        except Exception as e:
            print(e)
            logging.getLogger('runner').warning(e)
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
                        self.ev.add_summary_simple_value('validation-mean-{}-{}'.format(k, c), vv)

                    except Exception as e:
                        logging.getLogger('runner').warning('could not save {} as scalar summary value'.format(vv))

        return minerrors, avgerrors, medianerrors, maxerrors

    def test(self):
        self.ev.tedc.p = np.int32(self.ev.tedc.p)
        errors = self.ev.test_all_available(batch_size=1, testing=True)
        if self.results_to_csv and len(errors):
            minerrors, avgerrors, medianerrors, maxerrors = self.calc_min_mean_median_max_errors(errors)
            self.write_error_to_csv(errors, 'testing_scores.csv', minerrors, avgerrors, medianerrors, maxerrors)

    def run(self, **kw):
        # save this file as txt to cachefolder:

        shutil.copyfile(self.runfile, os.path.join(self.cachefolder, 'runfile.py'))

        if "train" in self.episodes:
            with self.ev.get_train_session() as sess:
                self.ev.set_session(sess, self.cachefolder, train=True)
                if self.checkpointfiles[0]:
                    self.ev.load(self.checkpointfiles[0])
                self.train()

        if "test" in self.episodes or "evaluate" in self.episodes:
            self.use_tensorboard = False  # no need, since we evaluate everything anyways.
            with self.ev.get_test_session() as sess:
                self.ev.set_session(sess, self.cachefolder)
                for est, ckpt in zip(self.estimatefilenames, self.checkpointfiles):
                    if ckpt:
                        self.ev.load(ckpt)
                    self.ev.estimatefilename = est
                    self.test()
        if self.notifyme:
            notify_user(self.notifyme['chat_id'], self.notifyme['token'],
                        message='{} has/have finished'.format(" and ".join(self.episodes)))


generate_defaults_info(Runner)
