__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import logging
logging.basicConfig(level=logging.INFO)
import os
import numpy as np
import sys
from options import get_args
from data.grid_collection import GridDataCollection, ThreadedGridDataCollection
from options.parser import clean_datacollection_args, clean_eval_args, clean_runner_args
from runner import Runner


def run_mdgru(args=None):
    """Executes a training/ testing or training and testing run for the mdgru network"""

    # Parse arguments
    fullparameters = " ".join(args if args is not None else sys.argv)
    args = get_args(args=args)

    # Set environment flag(s) and finally import the classes that depend upon them
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(g) for g in args.gpu])
    if args.use_pytorch:
        from model_pytorch.mdgru_classification import MDGRUClassification
        from eval.torch import SupervisedEvaluationTorch
        evalclass = SupervisedEvaluationTorch
    else:
        from model.mdgru_classification import MDGRUClassification
        from eval.tf import SupervisedEvaluationTensorflow
        evalclass = SupervisedEvaluationTensorflow
        from model.mdgru_classification import MDGRUClassification
        from model.mdgru_classification import MDGRUClassificationWithDiceLoss
        from model.mdgru_classification import MDGRUClassificationWithGeneralizedDiceLoss

    # Set the necessary classes
    dc = GridDataCollection
    tdc = dc if args.nonthreaded else ThreadedGridDataCollection
    if args.dice_generalized:
        if args.use_pytorch:
            raise Exception('diceloss is not yet implemented for pytorch')
        mclassification = MDGRUClassificationWithGeneralizedDiceLoss
    elif args.dice_loss_label != None or args.dice_autoweighted:
        if args.use_pytorch:
            raise Exception('diceloss is not yet implemented for pytorch')
        mclassification = MDGRUClassificationWithDiceLoss
    else:
        mclassification = MDGRUClassification
    if args.gpuboundfraction != 1:
        mclassification.set_allowed_gpu_memory_fraction(args.gpuboundfraction)

    # Infer the correct modelname and optionname (name to be used for experiments folder and evaluated files)
    modelname = None
    if args.modelname is not None:
        modelname = args.modelname
    elif args.ckpt is not None:
        modelname = mclassification.get_model_name_from_ckpt(args.ckpt[0])
    if modelname is None:
        modelname = "f{}m{}d{}-{}w{}p{}bn{}res{}lr{}r{}ns{}-de{}-{}se{}_all".format(int(1 - args.nofsg),
                                                                                "".join(args.m),
                                                                                args.droprate,
                                                                                args.nodropconnectx * 2 + args.nodropconnecth,
                                                                                "-".join([str(ww) for ww in args.w]),
                                                                                "-".join([str(pp) for pp in args.p]),
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

    # Set up datacollections
    args_tr, args_val, args_te = clean_datacollection_args(args)

    # Set up model and evaluation
    args_eval = clean_eval_args(args)

    args_eval["namespace"] = modelname

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

    ev = evalclass(mclassification, datadict,
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
