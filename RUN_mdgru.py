from helper import define_arguments

__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import logging
logging.basicConfig(level=logging.INFO)
import os
import numpy as np
import sys
from options.parser import get_parser
from data.grid_collection import GridDataCollection, ThreadedGridDataCollection
from options.parser import clean_datacollection_args, clean_eval_args, clean_runner_args
from runner import Runner
from helper import compile_arguments


def run_mdgru(args=None):
    """Executes a training/ testing or training and testing run for the mdgru network"""

    # Parse arguments
    fullparameters = " ".join(args if args is not None else sys.argv)
    parser = get_parser()
    pre_args, _ = parser.parse_known_args()
    parser.add_argument('-h','--help', action='store_true', help='print this help message')

    DROP_RATE_DEFAULT = 0.5
    # model_parameters = parser.add_argument_group('model parameters')

    MINMINIBATCH_DEFAULT = 16
    # model_parameters.add_argument('-d', '--droprate', default=DROP_RATE_DEFAULT, type=float,
    #                               help='drop rate used for dropconnect')
    # model_parameters.add_argument('--modelname', '--namespace', default=None, help='override modelname, probably not a good idea!')


    # args = get_args(args=args)

    # Set environment flag(s) and finally import the classes that depend upon them
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(g) for g in pre_args.gpu])
    if pre_args.use_pytorch:
        from model_pytorch.mdgru_classification import MDGRUClassification
        from eval.torch import SupervisedEvaluationTorch
        evalcls = SupervisedEvaluationTorch
    else:
        from model.mdgru_classification import MDGRUClassification
        from eval.tf import SupervisedEvaluationTensorflow
        evalcls = SupervisedEvaluationTensorflow

    # Set the necessary classes
    dc = GridDataCollection
    tdc = dc if pre_args.nonthreaded else ThreadedGridDataCollection
    modelcls = MDGRUClassification
    if pre_args.gpuboundfraction != 1:
        modelcls.set_allowed_gpu_memory_fraction(pre_args.gpuboundfraction)

    define_arguments(modelcls, parser.add_argument_group('model parameters'))
    define_arguments(evalcls, parser.add_argument_group('eval parameters'))

    args = parser.parse_args(args=args)
    if args.help:
        parser.print_help()
        return
    if args.optionname is not None:
        optionname = args.optionname
    else:
        optionname = 'default' #TODO: find better default model description! hash?

    # Set up datacollections
    args_tr, args_val, args_te = clean_datacollection_args(args)

    # Set up model and evaluation
    args_eval = clean_eval_args(args)
    args_model, _ = compile_arguments(modelcls, args)
    args_eval.update(args_model)
    if args_tr is not None:
        traindc = tdc(**args_tr)
    if args_val is not None:
        valdc = dc(**args_val)
    if args_te is not None:
        testdc = dc(**args_te)
    if args.onlytest: #FIXME: this is not the smartest way of doing it, make sure that you can allow missing entries in this dict!
        datadict = {"train": testdc, "validation": testdc, "test": testdc}
    elif args.onlytrain:
        datadict = {"train": traindc, "validation": valdc, "test": valdc}
    else:
        datadict = {"train": traindc, "validation": valdc, "test": testdc}

    ev = evalcls(modelcls, datadict,
                                             args_eval)

    # Set up runner
    args_runner = clean_runner_args(args)
    args_runner["fullparameters"] = fullparameters
    args_runner["estimatefilenames"] = optionname
    if isinstance(optionname, list):
        pf = "-".join(optionname)
        if len(pf) > 40:
            pf = pf[:39] + "..."
    else:
        pf = optionname
    args_runner["experiments_postfix"] = "_" + pf
    runner = Runner(ev, **args_runner)

    # Run computation
    return runner.run()


if __name__ == "__main__":
    run_mdgru()
