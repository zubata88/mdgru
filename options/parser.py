import argparse
import os
import numpy as np
import copy
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


def clean_datacollection_args(args):

    # w = np.asarray(args.windowsize)
    # if args.windowsizetesting:
    #     wt = np.asarray(args.windowsizetesting)
    # else:
    #     wt = w
    #
    # if args.windowsizevalidation:
    #     raise NotImplementedError('Unfortunately, windowsizevalidation can not yet be used!')
    #     wv = np.asarray(args.windowsizevalidation)
    # else:
    #     wv = w
    #
    # p = np.asarray(args.padding)
    # if args.paddingtesting:
    #     pt = np.asarray(args.paddingtesting)
    # else:
    #     pt = p
    #
    # if args.paddingvalidation:
    #     raise NotImplementedError('Unfortunately, paddingvalidation can not yet be used!')
    #     pv = np.asarray(args.paddingvalidation)
    # else:
    #     pv = p
    args_data = {
        "featurefiles": args.features,
        "maskfiles": args.mask,
        "seed": args.seed,
        "nooriginal": args.nooriginal,
        "nclasses": args.nclasses,
        "subtractGauss": 1 - args.nofsg,
        "correct_nifti_orientation": 1 - args.ignore_nifti_header,
        "channels_last": not args.use_pytorch,
        "perform_one_hot_encoding": not args.use_pytorch,
    }
    if args.mask is not None:
        args_data["choose_mask_at_random"] = len(args.mask) > 1
    if args.gausssigma is not None:
        args_data['sigma'] = args.gausssigma

    args_tr = None
    args_val = None
    args_te = None
    if not args.only_test:
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
            "batchsize": args.batch_size,
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
    if not args.only_train:
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
