from helper import define_arguments

__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import logging
logging.basicConfig(level=logging.INFO)
import os
import numpy as np
import sys
from data.grid_collection import GridDataCollection, ThreadedGridDataCollection
from options.parser import clean_datacollection_args
from runner import Runner
from helper import compile_arguments
import argparse


def run_mdgru(args=None):
    """Executes a training/ testing or training and testing run for the mdgru network"""

    # Parse arguments
    fullparameters = " ".join(args if args is not None else sys.argv)

    parser = argparse.ArgumentParser(description="evaluate any data with given parameters", add_help=False)

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
    sampling_parameters.add_argument('-s', '--seed', default=1234, type=int,
                                     help="seed for sampling random states")
    # sampling_parameters.add_argument('-b', '--batchsize', type=int, default=BATCHSIZE_DEFAULT, help="batch size")
    sampling_parameters.add_argument('--testbatchsize', type=int, help="test batch size")
    sampling_parameters.add_argument('--each_with_labels', default=0, type=int,
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
    sampling_parameters.add_argument('--mirror', nargs="+", help='array: mirror boolean for each image dim')
    sampling_parameters.add_argument('--datainterpolation', type=int,
                                     help='interpolation order. default is spline interpolation, 2 is quadratic, 1 is linear and 0 is NN')
    sampling_parameters.add_argument('--num_threads', default=1, type=int,
                                     help='specify how many threads should be run for threaded version of griddatacollection')


    execution_parameters = parser.add_argument_group('execution parameters')

    execution_parameters.add_argument('--gpu', type=int, nargs="+", default=[0], help="set gpu")


    pre_parameter = parser.add_argument_group('Options changing parameter. Use together with --help')
    pre_parameter.add_argument('--use_pytorch', action='store_true', help='use experimental pytorch version. Only core functionality is provided')
    pre_args, _ = parser.parse_known_args()
    parser.add_argument('-h','--help', action='store_true', help='print this help message')

    # Set environment flag(s) and finally import the classes that depend upon them
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(g) for g in pre_args.gpu])
    if pre_args.use_pytorch:
        from model_pytorch.mdgru_classification import MDGRUClassification as modelcls
        from eval.torch import SupervisedEvaluationTorch as evalcls
    else:
        from model.mdgru_classification import MDGRUClassification as modelcls
        from eval.tf import SupervisedEvaluationTensorflow as evalcls

    # Set the necessary classes
    dc = GridDataCollection
    tdc = dc if pre_args.nonthreaded else ThreadedGridDataCollection

    define_arguments(modelcls, parser.add_argument_group('Model Parameters'))
    define_arguments(evalcls, parser.add_argument_group('Evaluation Parameters'))
    define_arguments(Runner, parser.add_argument_group('Runner Parameters'))
    args = parser.parse_args(args=args)

    if args.help:
        parser.print_help()
        return
    if not args.use_pytorch:

        if args.gpubound != 1:
            modelcls.set_allowed_gpu_memory_fraction(args.gpubound)

    # Set up datacollections

    args_tr, args_val, args_te = clean_datacollection_args(args)

    # Set up model and evaluation
    kw = vars(args)
    args_eval, _ = compile_arguments(evalcls, kw, True)
    args_model, _ = compile_arguments(modelcls, kw, True)
    args_eval.update(args_model)
    if not args.use_pytorch and args.checkpointfiles is not None:
        args_eval['namespace'] = modelcls.get_model_name_from_ckpt(args.checkpointfiles[0])

    if args_tr is not None:
        traindc = tdc(**args_tr)
    if args_val is not None:
        valdc = dc(**args_val)
    if args_te is not None:
        testdc = dc(**args_te)
    if args.only_test: #FIXME: this is not the smartest way of doing it, make sure that you can allow missing entries in this dict!
        datadict = {"train": testdc, "validation": testdc, "test": testdc}
    elif args.only_train:
        datadict = {"train": traindc, "validation": valdc, "test": valdc}
    else:
        datadict = {"train": traindc, "validation": valdc, "test": testdc}

    ev = evalcls(modelcls, datadict,
                                             args_eval)

    # Set up runner
    args_runner, _ = compile_arguments(Runner, kw, True)
    args_runner.update({
        "experimentloc": os.path.join(args.datapath, 'experiments'),
        "fullparameters": fullparameters,
        # "estimatefilenames": optionname
    })
    runner = Runner(ev, **args_runner)
    # Run computation
    return runner.run()


if __name__ == "__main__":
    run_mdgru()
