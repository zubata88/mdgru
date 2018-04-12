__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import logging
logging.basicConfig(level=logging.INFO)
import os
import argparse
import numpy as np
import sys
import copy

DROP_RATE_DEFAULT = 0.5
MINMINIBATCH_DEFAULT = 16
BATCHSIZE_DEFAULT = 1
EACH_WITH_LABELS_DEFAULT = 0
GPU_DEFAULT = [0]
EPOCHS_DEFAULT = 10
LEARNINGRATE_DEFAULT = 1.0
RANDOM_SEED_DEFAULT = 123456789
UNCERTAINTYTIMES_DEFAULT = 1
TEST_EACH_DEFAULT = 2500
SAVE_EACH_DEFAULT = 2500
GPUBOUNDFRACTION_DEFAULT = 1

fullparameters = " ".join(sys.argv)

parser = argparse.ArgumentParser(description="evaluate any data with given parameters")

data_parameters = parser.add_argument_group('data parameters')
data_parameters.add_argument('--datapath', help='path where training, validation and testing and experiments folder'
                                                + ' lie. Can be overridden for the data folders, '
                                                + 'if location* is provided as an absolute path', required=True)
data_parameters.add_argument('--locationtraining', nargs='+',
                             help='foldername for training location or all training folders')
data_parameters.add_argument('--locationtesting', nargs='+', help='foldername for testing or all ')
data_parameters.add_argument('--locationvalidation', nargs='+', help='foldername for validation or all ')
data_parameters.add_argument('-f', '--features', nargs='+', help="feature files to be used", required=True)
data_parameters.add_argument('-m', '--mask', nargs='+', help='mask or label files to be used')
data_parameters.add_argument('--notestingmask', action="store_true",
                             help='use if there are no mask files for the test set')
gauss_sigma_group = data_parameters.add_mutually_exclusive_group()
gauss_sigma_group.add_argument('--nofsg', action="store_true", help="dont add gauss subtracted files (added by default)")
gauss_sigma_group.add_argument('--gausssigma', type=float, nargs='+',
                             help="array of sigmas to use for gauss highpass filtering")
data_parameters.add_argument('--nooriginal', action="store_true", help="no original files (added by default)")
data_parameters.add_argument('--nclasses', type=int, help="set number of classes.", required=True)
data_parameters.add_argument('--ignore_nifty_header', action="store_true", help="ignores header information about orientation of the data")

sampling_parameters = parser.add_argument_group('sampling parameters')
sampling_parameters.add_argument('-w', '--windowsize', type=int, nargs='+', help='patch or subvolume size',
                                 required=True)
sampling_parameters.add_argument('--windowsizevalidation', type=int, nargs='+',
                                 help="special window size for validation")
sampling_parameters.add_argument('--windowsizetesting', type=int, nargs='+', help="special window size for testing")
sampling_parameters.add_argument('-p', '--padding', type=int, nargs='+',
                                 help='padding allowed when sampling subvolumes or patches', required=True)
sampling_parameters.add_argument('--paddingvalidation', type=int, nargs='+', help='special padding for validation')
sampling_parameters.add_argument('--paddingtesting', type=int, nargs='+', help='special padding for testing')
sampling_parameters.add_argument('-s', '--seed', default=RANDOM_SEED_DEFAULT, type=int,
                                 help="seed for sampling random states")
sampling_parameters.add_argument('-b', '--batchsize', type=int, default=BATCHSIZE_DEFAULT, help="batch size")
sampling_parameters.add_argument('--testbatchsize', type=int, help="test batch size")
sampling_parameters.add_argument('--each_with_labels', default=EACH_WITH_LABELS_DEFAULT, type=int,
                                 help="each n-th random sample must contain labeled content.' "
                                      + " if 0, no special sampling is performed")
sampling_parameters.add_argument('--deformation', type=int, nargs='+', help="deformation array")
sampling_parameters.add_argument('--deformSigma', type=float, nargs='+',
                                 help="standard deformation of low resolution deformation grid.")
sampling_parameters.add_argument('--nonthreaded', action="store_true",
                                 help="disallow threading during training to preload data before the processing")
sampling_parameters.add_argument('--nonlazy', action="store_true", help="throw everything into memory")
sampling_parameters.add_argument('--rotate', help="random rotate by")
sampling_parameters.add_argument('--scale', nargs="+", help='random scale by')
sampling_parameters.add_argument('--datainterpolation', type=int,
                                 help='interpolation order. default is spline interpolation, 2 is quadratic, 1 is linear and 0 is NN')
sampling_parameters.add_argument('--num_threads', default=1, type=int,
                                 help='specify how many threads should be run for threaded version of griddatacollection')

model_parameters = parser.add_argument_group('model parameters')
model_parameters.add_argument('-d', '--droprate', default=DROP_RATE_DEFAULT, type=float,
                              help='drop rate used for dropconnect')
model_parameters.add_argument('--bnx', action="store_true", help="batchnorm over input x in rnn")
model_parameters.add_argument('--bnh', action="store_true", help="batchnorm over input h in rnn")
model_parameters.add_argument('--bna', action="store_true", help="batchnorm in activation of ht")
model_parameters.add_argument('--bne', action="store_true", help="batchnorm after matmul after mdrnn")
model_parameters.add_argument('--minminibatch', type=int, default=MINMINIBATCH_DEFAULT,
                              help="batchnorm number of mini batches to average over")
model_parameters.add_argument('--resx', action="store_true", help="residual learning for each input slice x")
model_parameters.add_argument('--resh', action="store_true", help="residual learning for each prev output h")
model_parameters.add_argument('--mdgrures', action="store_true", help="residual learning for each mdrnn as a whole")
model_parameters.add_argument('--modelname', default=None, help='override modelname')
model_parameters.add_argument('--dontsumcgrus', action="store_true",
                              help="use rnn results individually instead of as a sum")
model_parameters.add_argument('--putrback', action="store_true", help="use original gru formulation")
model_parameters.add_argument('--model_seed', type=int, default=None, help='set a seed such that all models will create reproducible initializations')
model_parameters.add_argument('--legacy_cgru_addition', action="store_true", help='allows to load old models despite new code. Only use when you know what you`re doing')

execution_parameters = parser.add_argument_group('execution parameters')
execution_parameters.add_argument('--nodropconnecth', action="store_true", help="dropconnect on prev output")
execution_parameters.add_argument('--nodropconnectx', action="store_true", help="no dropconnect on input slice")
execution_parameters.add_argument('--use_dropconnect_on_state', action="store_true", help="apply dropconnect also on the weights that are used to form the proposal in CGRU")
execution_parameters.add_argument('--gpu', type=int, nargs="+", default=GPU_DEFAULT, help="set gpu")
execution_parameters.add_argument('--iterations', type=int, help="set num iterations, overrides epochs if set")
execution_parameters.add_argument('--epochs', type=int, default=EPOCHS_DEFAULT, help="set num epochs")
execution_parameters.add_argument('--onlytest', action="store_true",
                                                help="only perform testing phase, "
                                                     + "inferred, when no training and validation location is present")
execution_parameters.add_argument('--onlytrain', action="store_true",
                                                 help="only perform training phase, "
                                                      + "inferred, when no testing location is present")
execution_parameters.add_argument('--ckpt', default=None,
                                  help='provide checkpointfile for this template. If no modelname is provided, we will infer one from this file')
execution_parameters.add_argument('--learningrate', type=float, default=LEARNINGRATE_DEFAULT,
                                  help='learningrate (for adadelta)')
execution_parameters.add_argument('--optionname', help='override optionname (used eg for saving results)')

execution_parameters.add_argument('--testfirst', action="store_true", help="validate first")
execution_parameters.add_argument('--test_each', default=TEST_EACH_DEFAULT, type=int, help='validate each # iterations')
execution_parameters.add_argument('--save_each', default=SAVE_EACH_DEFAULT, type=int, help='save to ckpt each # iterations')
execution_parameters.add_argument('--gpuboundfraction', default=GPUBOUNDFRACTION_DEFAULT, type=float,
                                  help='manage how much of the memory of the gpu can be used')
execution_parameters.add_argument('--cpu', action="store_true", help='Only run on cpu')
execution_parameters.add_argument('--image_summaries_each', type=int, default=None,
                                  help='Specify, if tensorboard is used, how often we want to draw image summaries during training')
execution_parameters.add_argument('--dont_use_tensorboard', action="store_true", help='Deactivates tensorboard usage')
execution_parameters.add_argument('--swap_memory', action="store_true", help="Allows for larger volume sizes with a trade for some speed, moves data from cpu to gpu during one iteration")
execution_parameters.add_argument('--notifyme', default=None, nargs='?', type=str,
                                  help='Experimental feature that when something goes amiss, '
                                       + 'this telegram chat id will be used to inform the '
                                       + 'respective user of the error. This requires a file called config.json'
                                       + ' in the same folder as this file, containing a simple dict structure as'
                                       + ' follows: {"chat_id": CHATID, "token": TOKEN}, where CHATID and TOKEN '
                                       + 'have to be created with Telegrams BotFather. The chatid from config can be '
                                       + 'overriden using a parameter together with this option.')
execution_parameters.add_argument('--results_to_csv', action="store_true", help='Writes validation scores to validation_scores.csv')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(g) for g in args.gpu])
# THIS is here so we are sure, we can set cuda visible devices beforehand.
from data.grid_collection import GridDataCollection, ThreadedGridDataCollection
from model.classification import MDGRUClassification
from eval.classification import LargeVolumeClassificationEvaluation
from runner import Runner
import tensorflow as tf
if args.model_seed is not None:
    tf.set_random_seed(args.model_seed)

# compile all required arguments and get defaults for required arguments that are depending on others:
dc = GridDataCollection
tdc = dc if args.nonthreaded else ThreadedGridDataCollection

w = np.asarray(args.windowsize)
if args.windowsizetesting:
    wt = np.asarray(args.windowsizetesting)
else:
    wt = w

if args.windowsizevalidation:
    raise NotImplementedError('Unfortunately, windowsizevalidation can not yet be used!')
    wv = np.asarray(args.windowsizevalidation)
else:
    wv = w

p = np.asarray(args.padding)
if args.paddingtesting:
    pt = np.asarray(args.paddingtesting)
else:
    pt = p

if args.paddingvalidation:
    raise NotImplementedError('Unfortunately, paddingvalidation can not yet be used!')
    pv = np.asarray(args.paddingvalidation)
else:
    pv = p

seed = args.seed
f = args.features
m = args.mask
d = args.droprate
mt = [] if args.notestingmask else m
nclasses = args.nclasses

args_runner = {
    "epochs": 0 if args.iterations else args.epochs,
    "test_size": 1,
    "test_iters": 1,
    "show_testing_results": True,
    "perform_n_times_full_validation": 1,
    "perform_n_times_full_validation_dropout": 1,
    "experimentloc": os.path.join(args.datapath, 'experiments'),
    "fullparameters": fullparameters,
}

if args.results_to_csv:
    args_runner["results_to_csv"] = True

if args.notifyme:
    try:
        import json
        data = json.load(open('config.json'))
        nm = dict(chat_id=data['chat_id'], token=data['token'])
        try:
            nm['chat_id'] = int(args.notifyme)
        except Exception:
            pass
        finally:
            args_runner['notifyme'] = nm
    except:
        # we give up
        print('notifyme id not understood')

minminibatch = args.minminibatch
if minminibatch < 2:
    minminibatch = None

modelname = None
if args.modelname is not None:
    modelname = args.modelname
elif args.ckpt is not None:
    from tensorflow.python import pywrap_tensorflow
    try:
        r = pywrap_tensorflow.NewCheckpointReader(args.ckpt)
        modelname = r.get_variable_to_shape_map().popitem()[0].split('/')[0]
    except:
        logging.getLogger('runfile').warning('could not load modelname from ckpt-file {}'.format(args.ckpt))
if modelname is None:
    modelname = "f{}m{}d{}-{}w{}p{}bn{}res{}lr{}r{}ns{}-de{}-{}se{}_all".format(int(1 - args.nofsg),
                                                                            "".join(m),
                                                                            d,
                                                                            args.nodropconnectx * 2 + args.nodropconnecth,
                                                                            "-".join([str(ww) for ww in w]),
                                                                            "-".join([str(pp) for pp in p]),
                                                                            int(args.bnx) * 8 + int(args.bnh) * 4 + int(
                                                                                args.bna) * 2 + int(args.bne) * 1,
                                                                            int(args.resx) * 4 + int(
                                                                                args.resh) * 2 + int(args.mdgrures) * 1,
                                                                            args.learningrate,
                                                                            args.putrback,
                                                                            args.dontsumcgrus,
                                                                            args.deformSigma if np.isscalar(args.deformSigma)
                                                                                or args.deformSigma is None else "_".join(
                                                                                ["{}".format(x) for x in
                                                                                 args.deformSigma]),
                                                                            "-".join(
                                                                                [str(dd) for dd in args.deformation]),
                                                                            args.each_with_labels)

if args.optionname is not None:
    optionname = args.optionname
else:
    optionname = modelname

if args.gpuboundfraction != 1:
    tf.GPUOptions(per_process_gpu_memory_fraction=args.gpuboundfraction)
rnn_activation = tf.nn.tanh
activation = tf.nn.tanh
# now move all optional parameters into dicts that can be passed to their respective functions!

# runner arguments
if args.onlytest or (args.locationtraining is None and args.locationvalidation is None):
    args_runner['episodes'] = ['test']
elif args.onlytrain or (args.locationtesting is None):
    args_runner['episodes'] = ['train']
if args.iterations is not None:
    args_runner['its_per_epoch'] = args.iterations
if args.ckpt is not None:
    args_runner['checkpointfile'] = args.ckpt
if args.testbatchsize is not None:
    args_runner['test_size'] = args.testbatchsize
if args.test_each is not None:
    args_runner['test_each'] = args.test_each
if args.save_each is not None:
    args_runner['save_each'] = args.save_each
if args.gpuboundfraction is not None:
    args_runner['gpubound'] = args.gpuboundfraction
if args.testfirst:
    args_runner['test_first'] = args.testfirst
if args.cpu is not None:
    args_runner['only_cpu'] = args.cpu

# data arguments

args_data = {
    "seed": seed,
    "nooriginal": args.nooriginal,
    "nclasses": nclasses,
    "subtractGauss": 1 - args.nofsg,
    "correct_nifti_orientation": 1 - args.ignore_nifty_header
}
if m is not None:
    args_data["choose_mask_at_random"] = len(m) > 1
if args.gausssigma is not None:
    args_data['sigma'] = args.gausssigma

if not args.onlytest:
    if len(args.locationtraining) > 1:
        locationtraining = None
        tpstraining = args.locationtraining
    else:
        locationtraining = os.path.join(args.datapath, args.locationtraining[0])
        tpstraining = None
    args_tr = {
        "location": locationtraining,
        "tps": tpstraining,
        "w": w,
        "padding": p,
        "lazy": (False == args.nonlazy),
        "batchsize": args.batchsize,
    }
    if args.deformation is not None:
        args_tr['deformation'] = args.deformation
    if args.deformSigma is not None:
        args_tr['deformSigma'] = args.deformSigma
    if args.each_with_labels is not None:
        args_tr['each_with_labels'] = args.each_with_labels
    if args.rotate is not None:
        args_tr['rotation'] = args.rotate
    if args.scale is not None:
        args_tr['scaling'] = args.scale
    if args.datainterpolation is not None:
        args_tr['datainterpolation'] = args.datainterpolation
    if args.num_threads is not None:
        args_tr['num_threads'] = args.num_threads
    args_tr.update(copy.deepcopy(args_data))
    traindc = tdc(f, m, **args_tr)

    if len(args.locationvalidation) > 1:
        locationvalidation = None
        tpsvalidation = args.locationvalidation
    else:
        locationvalidation = os.path.join(args.datapath, args.locationvalidation[0])
        tpsvalidation = None
    args_val = {
        "location": locationvalidation,
        "tps": tpsvalidation,
        "w": wv,
        "padding": pv,
    }
    args_val.update(copy.deepcopy(args_data))
    valdc = dc(f, m, **args_val)
else:
    traindc = None
    testdc = None
if not args.onlytrain:
    if len(args.locationtesting) > 1:
        locationtesting = None
        tpstesting = args.locationtesting
    else:
        locationtesting = os.path.join(args.datapath, args.locationtesting[0])
        tpstesting = None
    args_te = {
        "location": locationtesting,
        "tps": tpstesting,
        "w": wt,
        "padding": pt,
    }
    args_te.update(copy.deepcopy(args_data))
    testdc = dc(f, mt, **args_te)
else:
    testdc = None
# eval and model arguments

args_eval = {"batch_size": args.batchsize,
             "learning_rate": args.learningrate,
             "w": w,
             "whiten": False,
             "min_mini_batch": minminibatch,
             "dropout_rate": d,
             "use_dropconnecth": 1 - args.nodropconnecth,
             "use_dropconnectx": 1 - args.nodropconnectx,
             "namespace": modelname,
             "nclasses": nclasses,
             'rnn_activation': rnn_activation,
             'activation': activation,
             'validate_same': True,
             'use_tensorboard': 1-args.dont_use_tensorboard,
             'swap_memory': args.swap_memory,
             'use_dropconnect_on_state': args.use_dropconnect_on_state,
             'legacy_cgru_addition': args.legacy_cgru_addition,
             }

if not args.dont_use_tensorboard and args.image_summaries_each is not None:
    args_eval['image_summaries_each'] = args.image_summaries_each
argvars = vars(args)
arglookup = {
    'learning_rate': 'learningrate',
    'batch_size': 'batchsize',
    'bnx': 'bnx', 'bnh': 'bnh', 'bne': 'bne', 'bna': 'bna',
    'resmdgru': 'mdgrures',
    'resgrux': 'resx',
    'resgruh': 'resh',
    'return_cgru_results': 'dontsumcgrus',
    'put_r_back': 'putrback',
}
mclassification = MDGRUClassification
for k, v in arglookup.items():
    if v in argvars.keys():
        args_eval[k] = argvars[v]

if args.onlytest: #FIXME: this is not the smartest way of doing it, make sure that you can allow missing entries in this dict!
    datadict = {"train": testdc, "validation": testdc, "test": testdc}
elif args.onlytrain:
    datadict = {"train": traindc, "validation": valdc, "test": valdc}
else:
    datadict = {"train": traindc, "validation": valdc, "test": testdc}

ev = LargeVolumeClassificationEvaluation(mclassification, datadict,
                                         **args_eval)
ev.estimatefilename = optionname
Runner(ev, experiments_postfix="_" + optionname, **args_runner).run()
