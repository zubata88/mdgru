from mdgru.helper import define_arguments

__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import logging
logging.basicConfig(level=logging.INFO)
import os
import numpy as np
import sys
from mdgru.data.grid_collection import GridDataCollection, ThreadedGridDataCollection
# from options.parser import clean_datacollection_args
from mdgru.runner import Runner
from mdgru.helper import compile_arguments
import argparse


def run_mdgru(args=None):
    """Executes a training/ testing or training and testing run for the mdgru network"""

    # Parse arguments
    fullparameters = " ".join(args if args is not None else sys.argv)

    parser = argparse.ArgumentParser(description="evaluate any data with given parameters", add_help=False)

    pre_parameter = parser.add_argument_group('Options changing parameter. Use together with --help')
    pre_parameter.add_argument('--use_pytorch', action='store_true', help='use experimental pytorch version. Only core functionality is provided')
    pre_parameter.add_argument('--gpu', type=int, nargs='+', default=[0], help='set gpu ids')
    pre_parameter.add_argument('--nonthreaded', action="store_true",
                                     help="disallow threading during training to preload data before the processing")
    pre_parameter.add_argument('--dice_loss_label', default=None, type=int, nargs="+", help='labels for which the dice losses shall be calculated')
    pre_parameter.add_argument('--dice_loss_weight', default=None, type=float, nargs="+", help='weights for the dice losses of the individual classes. same size as dice_loss_label or scalar if dice_autoweighted. final loss: sum(dice_loss_weight)*diceloss + (1-sum(dice_loss_weight))*crossentropy')
    pre_parameter.add_argument('--dice_autoweighted', action="store_true", help='weights the label Dices with the squared inverse gold standard area/volume; specify which labels with dice_loss_label; sum(dice_loss_weight) is used as a weighting between crossentropy and diceloss')
    pre_parameter.add_argument('--dice_generalized', action="store_true", help='total intersections of all labels over total sums of all labels, instead of linearly combined class Dices')
    pre_parameter.add_argument('--dice_cc', action='store_true', help='dice loss for binary segmentation per true component')
    pre_args, _ = parser.parse_known_args(args=args)
    parser.add_argument('-h','--help', action='store_true', help='print this help message')

    # Set environment flag(s) and finally import the classes that depend upon them
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(g) for g in pre_args.gpu])
    if pre_args.use_pytorch:
        if pre_args.dice_cc:
            from mdgru.model_pytorch.mdgru_classification import MDGRUClassificationCC as modelcls
        else:
            from mdgru.model_pytorch.mdgru_classification import MDGRUClassification as modelcls
        from mdgru.eval.torch import SupervisedEvaluationTorch as evalcls
    else:
        if pre_args.dice_generalized:
            from mdgru.model.mdgru_classification import MDGRUClassificationWithGeneralizedDiceLoss as modelcls
        elif pre_args.dice_loss_label != None or pre_args.dice_autoweighted:
            from mdgru.model.mdgru_classification import MDGRUClassificationWithDiceLoss as modelcls
        else:
            from mdgru.model.mdgru_classification import MDGRUClassification as modelcls
        from mdgru.eval.tf import SupervisedEvaluationTensorflow as evalcls


    # Set the necessary classes
    # dc = GridDataCollection
    tdc = GridDataCollection if pre_args.nonthreaded else ThreadedGridDataCollection

    define_arguments(modelcls, parser.add_argument_group('Model Parameters'))
    define_arguments(evalcls, parser.add_argument_group('Evaluation Parameters'))
    define_arguments(Runner, parser.add_argument_group('Runner Parameters'))
    define_arguments(tdc, parser.add_argument_group('Data Parameters'))
    args = parser.parse_args(args=args)
    print(args)

    if args.help:
        parser.print_help()
        return
    if not args.use_pytorch:
        if args.gpubound != 1:
            modelcls.set_allowed_gpu_memory_fraction(args.gpubound)


    # Set up datacollections

    # args_tr, args_val, args_te = clean_datacollection_args(args)

    # Set up model and evaluation
    kw = vars(args)
    args_eval, _ = compile_arguments(evalcls, kw, True, keep_entries=True)
    args_model, _ = compile_arguments(modelcls, kw, True, keep_entries=True)
    args_data, _ = compile_arguments(tdc, kw, True, keep_entries=True)
    args_eval.update(args_model)
    args_eval.update(args_data)
    if not args.use_pytorch:
        if args.checkpointfiles is not None:
            args_eval['namespace'] = modelcls.get_model_name_from_ckpt(args.checkpointfiles[0])
    args_eval['channels_first'] = args.use_pytorch


    # if args_tr is not None:
    #     traindc = tdc(**args_tr)
    # if args_val is not None:
    #     valdc = tdc(**args_val)
    # if args_te is not None:
    #     testdc = dc(**args_te)
    # if args.only_test: #FIXME: this is not the smartest way of doing it, make sure that you can allow missing entries in this dict!
    #     datadict = {"train": testdc, "validation": testdc, "test": testdc}
    # elif args.only_train:
    #     datadict = {"train": traindc, "validation": valdc, "test": valdc}
    # else:
    #     datadict = {"train": traindc, "validation": valdc, "test": testdc}

    ev = evalcls(modelcls, tdc, args_eval)

    # Set up runner
    args_runner, _ = compile_arguments(Runner, kw, True, keep_entries=True)
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
