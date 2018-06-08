import argparse
import os
import numpy as np
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


def get_parser():
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
    data_parameters.add_argument('--ignore_nifti_header', action="store_true", help="ignores header information about orientation of the data")

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
    sampling_parameters.add_argument('--truncated_deform', action="store_true", help="deformations with displacements of maximum 3 times gausssigma in each spatial direction")

    sampling_parameters.add_argument('--nonthreaded', action="store_true",
                                     help="disallow threading during training to preload data before the processing")
    sampling_parameters.add_argument('--nonlazy', action="store_true", help="throw everything into memory")
    sampling_parameters.add_argument('--rotate', help="random rotate by")
    sampling_parameters.add_argument('--scale', nargs="+", help='random scale by')
    sampling_parameters.add_argument('--mirror', nargs="+", help='array: mirror boolean for each image dim')
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
    model_parameters.add_argument('--filter_size_x', default=None, type=int, nargs="+", help='filter sizes for each dimension for the input')
    model_parameters.add_argument('--filter_size_h', default=None, type=int, nargs="+", help='filter sizes for each dimension for the previous output')
    model_parameters.add_argument('--dice_loss_label', default=None, type=int, nargs="+", help='labels for which the dice loss shall be calculated')
    model_parameters.add_argument('--dice_loss_weight', default=None, type=float, nargs="+", help='dice loss weight combined with (1-sum(weight))*crossentropy')
    model_parameters.add_argument('--dice_autoweighted', action="store_true", help='weights the label Dices with the squared inverse gold standard area/volume; specify over which labels with dice_loss_label; sum of dice_loss_weight is used a weighting between cross entropy and generalized dice')
    model_parameters.add_argument('--dice_generalized', action="store_true", help='total intersections of all labels over total sums of all labels, instead of summed Dices')

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
    execution_parameters.add_argument('--ckpt', default=None, nargs="+",
                                      help='provide checkpointfile for this template. If no modelname is provided, '
                                           'we will infer one from this file. Multiple files are only allowed if '
                                           'only_test is set. If the same number of optionnames are provided, they will '
                                           'be used to name the different results. If only one optionname is provided, they '
                                           'will be numbered in order and the checkpoint filename will be included in the '
                                           'result file.')
    execution_parameters.add_argument('--learningrate', type=float, default=LEARNINGRATE_DEFAULT,
                                      help='learningrate (for adadelta)')
    execution_parameters.add_argument('--optionname', nargs="+", help='override optionname (used eg for saving results). '
                                                                      'Can be either 1, or n in the case of n ckpts ')

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
    execution_parameters.add_argument('--only_save_labels', action='store_true', help='Writes only labelmaps to disk, ignoring'
                                                                                      'probability maps and hence saving a lot of disk space.')
    execution_parameters.add_argument('--results_to_csv', action="store_true", help='Writes validation scores to validation_scores.csv')
    execution_parameters.add_argument('--number_of_evaluation_samples', type=int, default=1,
                                      help='Number times we want to evaluate one volume. This only makes sense '
                                           'using a keep rate of less than 1 during evaluation (dropout_during_evaluation '
                                           'less than 1)')
    execution_parameters.add_argument('--dropout_during_evaluation', type=float, default=1.0,
                                      help='Keeprate of weights during evaluation. Useful to visualize uncertainty '
                                           'in conjunction with a number of samples per volume')
    execution_parameters.add_argument('--save_individual_evaluations', action='store_true',
                                      help='Save each evaluation sample per volume. Without this flag, only the '
                                           'standard deviation and mean over all samples is kept.')

    return parser


def clean_datacollection_args(args):

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
    args_data = {
        "featurefiles": args.features,
        "maskfiles": args.mask,
        "seed": args.seed,
        "nooriginal": args.nooriginal,
        "nclasses": args.nclasses,
        "subtractGauss": 1 - args.nofsg,
        "correct_nifti_orientation": 1 - args.ignore_nifti_header
    }
    if args.mask is not None:
        args_data["choose_mask_at_random"] = len(args.mask) > 1
    if args.gausssigma is not None:
        args_data['sigma'] = args.gausssigma

    args_tr = None
    args_val = None
    args_te = None
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
        if args.truncated_deform is not None:
            args_tr['truncated_deform'] = args.truncated_deform
        if args.each_with_labels is not None:
            args_tr['each_with_labels'] = args.each_with_labels
        if args.rotate is not None:
            args_tr['rotation'] = args.rotate
        if args.scale is not None:
            args_tr['scaling'] = args.scale
        if args.mirror is not None:
            args_tr['mirror'] = args.mirror
        if args.datainterpolation is not None:
            args_tr['datainterpolation'] = args.datainterpolation
        if args.num_threads is not None:
            args_tr['num_threads'] = args.num_threads
        args_tr.update(copy.deepcopy(args_data))

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
        args_te["maskfiles"] = [] if args.notestingmask else args.mask

    return args_tr, args_val, args_te


def clean_eval_args(args):
    def harmonize_filter_size(fs, w):
        if fs is None:
            return [7 for _ in w]
        if len(fs) != len(w):
            if len(fs) == 1:
                fs = [fs[0] for _ in w]
            elif fs is None:
                fs = [7 for _ in w]
            else:
                print('Filter size and number of dimensions for subvolume do not match!')
                exit(0)
        return fs


    filter_size_x = harmonize_filter_size(args.filter_size_x, args.windowsize)
    filter_size_h = harmonize_filter_size(args.filter_size_h, args.windowsize)

    args_eval = {"batch_size": args.batchsize,
                 "learning_rate": args.learningrate,
                 "whiten": False,
                 "min_mini_batch": None if args.minminibatch < 2 else args.minminibatch,
                 "dropout_rate": args.droprate,
                 "use_dropconnect_h": 1 - args.nodropconnecth,
                 "use_dropconnect_x": 1 - args.nodropconnectx,
                 "nclasses": args.nclasses,
                 'validate_same': True,
                 'use_tensorboard': 1-args.dont_use_tensorboard,
                 'swap_memory': args.swap_memory,
                 'use_dropconnect_on_state': args.use_dropconnect_on_state,
                 'legacy_cgru_addition': args.legacy_cgru_addition,
                 'evaluate_uncertainty_times': args.number_of_evaluation_samples,
                 'evaluate_uncertainty_dropout': args.dropout_during_evaluation,
                 'evaluate_uncertainty_saveall': args.save_individual_evaluations,
                 'only_save_labels': args.only_save_labels,
                 'filter_size_x': filter_size_x,
                 'filter_size_h': filter_size_h,
                 'model_seed': args.model_seed,
                 "dice_loss_label": args.dice_loss_label,
                 "dice_loss_weight": args.dice_loss_weight,
                 "dice_autoweighted": args.dice_autoweighted,
                 "dice_generalized": args.dice_generalized,
                 }

    if not args.dont_use_tensorboard and args.image_summaries_each is not None:
        args_eval['image_summaries_each'] = args.image_summaries_each
    argvars = vars(args)
    arglookup = {
        'learning_rate': 'learningrate',
        'batch_size': 'batchsize',
        'add_x_bn': 'bnx', 'add_h_bn': 'bnh', 'add_e_bn': 'bne', 'add_a_bn': 'bna',
        'resmdgru': 'mdgrures',
        'resgrux': 'resx',
        'resgruh': 'resh',
        'return_cgru_results': 'dontsumcgrus',
        'put_r_back': 'putrback',
    }
    for k, v in arglookup.items():
        if v in argvars.keys():
            args_eval[k] = argvars[v]
    if args.cpu is not None:
        args_eval['only_cpu'] = args.cpu
    if args.gpuboundfraction is not None:
        args_eval['gpubound'] = args.gpuboundfraction
    return args_eval


def clean_runner_args(args):
    args_runner = {
        "epochs": 0 if args.iterations else args.epochs,
        "test_size": 1,
        "test_iters": 1,
        "show_testing_results": True,
        "perform_n_times_full_validation": 1,
        "perform_n_times_full_validation_dropout": 1,
        "experimentloc": os.path.join(args.datapath, 'experiments'),
        "results_to_csv": args.results_to_csv,
    }

    if args.notifyme:
        try:
            import json
            data = json.load(open('../config.json'))
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
    if args.onlytest or (args.locationtraining is None and args.locationvalidation is None):
        args_runner['episodes'] = ['test']
    elif args.onlytrain or (args.locationtesting is None):
        args_runner['episodes'] = ['train']
    if args.iterations is not None:
        args_runner['its_per_epoch'] = args.iterations
    if args.ckpt is not None:
        args_runner['checkpointfiles'] = args.ckpt
    if args.testbatchsize is not None:
        args_runner['test_size'] = args.testbatchsize
    if args.test_each is not None:
        args_runner['test_each'] = args.test_each
    if args.save_each is not None:
        args_runner['save_each'] = args.save_each
    if args.testfirst:
        args_runner['test_first'] = args.testfirst

    return args_runner
