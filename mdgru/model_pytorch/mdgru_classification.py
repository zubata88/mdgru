__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import numpy as np

from mdgru.helper import argget, compile_arguments
from mdgru.model_pytorch.mdrnn import MDGRUBlock
from mdgru.model_pytorch.mdrnn.mdgru import MDRNN
from mdgru.helper import collect_parameters, define_arguments
from . import ClassificationModel
import torch as th
from mdgru.model_pytorch import init_weights
import torch.nn.functional as F
from scipy.ndimage.measurements import label


class MDGRUClassification(ClassificationModel):
    """ Provides a full MDGRU default network.

    Using the parameters fc_channels and mdgru_channels, which have to be lists of the same length, a
    MDGRU network of a number of mdgru and voxel-wise fully connected layers can be generated. Strides can be
    set for each MDGRU layer as a list of lists of strides per dimension. Furthermore, entries in fc_channels may be
    None, when MDGRU layer should be stacked directly after each other.

    :param fc_channels: Defines the number of channels per voxel-wise fully connected layer
    :param mdgru_channels: Defines the number of channels per MDGRU layer
    :param strides: list of list defining the strides per dimension per MDGRU layer. None means strides of 1
    :param data_shape: subvolume size
    """

    def __init__(self, data_shape, dropout, kw):
        super(MDGRUClassification, self).__init__(data_shape, dropout, kw)
        my_kw, kw = compile_arguments(MDGRUClassification, kw, transitive=False)
        for k, v in my_kw.items():
            setattr(self, k, v)
        self.fc_channels = argget(kw, "fc_channels", [25, 45, self.nclasses])
        self.mdgru_channels = argget(kw, "mdgru_channels", [16, 32, 64])
        self.strides = argget(kw, "strides", [None for _ in self.mdgru_channels])
        self.data_shape = data_shape
        # create logits:
        logits = []
        num_spatial_dims = len(data_shape[2:])
        last_output_channel_size = data_shape[1]
        for it, (mdgru, fcc, s) in enumerate(zip(self.mdgru_channels, self.fc_channels, self.strides)):
            mdgru_kw = {}
            mdgru_kw.update(kw)
            if it == len(self.mdgru_channels) - 1:
                mdgru_kw["noactivation"] = True
            if s is not None:
                mdgru_kw["strides"] = [s for _ in range(num_spatial_dims)] if np.isscalar(s) else s
            logits += [MDGRUBlock(num_spatial_dims, self.dropout, last_output_channel_size, mdgru, fcc, mdgru_kw)]
            last_output_channel_size = fcc if fcc is not None else mdgru
        self.model = th.nn.Sequential(*logits)
        self.losses = (th.nn.modules.CrossEntropyLoss())
        print(self.model)

    def prediction(self, batch):
        """Provides prediction in the form of a discrete probability distribution per voxel"""
        pred = F.softmax(self.model(batch))
        return pred

    def initialize(self):
        self.model.apply(init_weights)

    @staticmethod
    def collect_parameters():
        args = collect_parameters(MDGRUBlock, {})
        args = collect_parameters(MDRNN, args)
        args = collect_parameters(MDRNN._defaults['crnn_class']['value'], args)
        return args

    @staticmethod
    def compile_arguments(kw, keep_entries=True):
        block_kw, kw = compile_arguments(MDGRUBlock, kw, transitive=True, keep_entries=keep_entries)
        mdrnn_kw, kw = compile_arguments(MDRNN, kw, transitive=True, keep_entries=keep_entries)
        crnn_kw, kw = compile_arguments(MDRNN._defaults['crnn_class']['value'], kw, transitive=True,
                                        keep_entries=keep_entries)
        new_kw = {}
        new_kw.update(crnn_kw)
        new_kw.update(mdrnn_kw)
        new_kw.update(block_kw)
        return new_kw, kw


class MDGRUClassificationCC(MDGRUClassification):
    _defaults = {'dice_loss_weight': {'value': 0, 'help': 'dice loss weights to be used'}}

    # 'use_connected_component_dice_loss': {'value': False, 'help': 'Use connected component dice loss, needs connected component labelling in griddatacollection to be performed. experimental', 'type': int}}

    def __init__(self, data_shape, dropout, kw):
        super(MDGRUClassificationCC, self).__init__(data_shape, dropout, kw)
        # self.dice_loss_label = argget(kw, "dice_loss_label", [])
        # self.dice_loss_weight = argget(kw, "dice_loss_weight", []) #here, this should contain one value!
        my_kw, kw = compile_arguments(MDGRUClassificationCC, kw, transitive=False)
        for k, v in my_kw.items():
            setattr(self, k, v)
        self.ce = th.nn.modules.CrossEntropyLoss()

    def losses(self, prediction, labels):
        pred = F.softmax(prediction)
        eps = 1e-8
        tp = th.FloatTensor([0])
        if prediction.is_cuda:
            tp = tp.cuda()
        nlabs = labels.max().cpu().item()
        for i in range(nlabs):
            mask = labels == i + 1
            tp += th.sum(th.masked_select(pred[:, 1], mask)) / th.sum(mask).float() / (nlabs + 1)
        mask = labels == 0
        fp = th.sum(th.masked_select(pred[:, 1], mask)) / th.sum(mask).float()
        if (nlabs == 0):
            diceLoss = th.zeros_like(tp)
        else:
            diceLoss = 1 - 2 * tp / (tp + 1 + fp)
        mask = (labels > 0).long()

        #compute my lesion dice:
        with th.no_grad():
            nppred = pred[:, 1].detach().cpu().numpy()
            vals = np.unique(nppred)
            vals = list(vals)
            if vals[0] > 0:
                vals.insert(0, 0)
            if vals[-1] < 1:
                vals.append(1)
            refmask = labels
            diceLossnp = 0
            print(vals)
            for it in range(len(vals)-1):
                segmask, numsegs = label(nppred >= vals[it+1])
                # overlap = False
                segs = [True for i in range(numsegs)]
                segs[0] = False
                diceLosst = []
                for ti in range(1, nlabs+1):
                    overlap = False
                    for si in range(1, numsegs):
                        #check connection:
                        d = 2*np.sum((label==ti)*(segmask==si))/(np.sum(label==ti)+np.sum(segmask==si))
                        if (d > 0):
                            diceLosst.append(d)
                            if si in segs:
                                segs[si] = False
                            # diceLossNum += 1
                            overlap = True
                    if not overlap:
                        diceLosst += [0]
                diceLosst += [0 for _ in range(np.sum(segs))]
                diceLossnp += np.mean(diceLosst)*(vals[it+1]-vals[it])
                # print(diceLossnp, diceLosst)
            print(diceLosst)


        return float(np.sum(self.dice_loss_weight)) * diceLoss, float((1 - np.sum(self.dice_loss_weight))) * self.ce(
            prediction, mask)
