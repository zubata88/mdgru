__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import torch as th
from mdgru.helper import argget, generate_defaults_info
from mdgru.helper import compile_arguments
from .mdgru import MDRNN


class MDGRUBlock(th.nn.Module):
    """Convenience class combining attributes to be used for multiple MDRNN and voxel-wise fully connected layers.
        :param num_spatial_dims: Number of spatial dimensions to consider
        :param dropout: Dropout rate provided as "keep" rate
        :param num_input: Nuber of input units/channels
        :param num_hidden: number of hidden units, output units of the MDRNN
        :param num_output: number of output units of the voxel-wise fully connected layer
                           (Can be None -> no voxel-wise fully connected layer)
    """
    _defaults = {
        "resmdgru": {'value': False, 'help': 'Add a residual connection from each mdgru input to its output, possibly homogenizing dimensions using one 1 conv layer'},
        "vwfc_activation": th.nn.Tanh,
        "noactivation": False,
        "name": None,
    }

    def __init__(self, num_spatial_dims, dropout, num_input, num_hidden, num_output, kw):
        super(MDGRUBlock, self).__init__()
        mdrnn_net_kw, kw = compile_arguments(MDGRUBlock, kw, transitive=False)
        for k, v in mdrnn_net_kw.items():
            setattr(self, k, v)
        self.mdrnn_kw, kw = compile_arguments(MDRNN, kw, transitive=True)
        self.crnn_kw, kw = compile_arguments(self.mdrnn_kw['crnn_class'], kw, transitive=True)

        spatial_dimensions = argget(kw, "dimensions", None)
        if spatial_dimensions is None:
            spatial_dimensions = [i for i in range(num_spatial_dims)]
        mdrnn_kw = {}
        mdrnn_kw.update(self.mdrnn_kw)
        mdrnn_kw.update(self.crnn_kw)
        mdrnn_kw.update(kw)

        mdrnn_kw["num_hidden"] = num_hidden
        mdrnn_kw["num_input"] = num_input
        mdrnn_kw["name"] = "mdgru"
        model = [MDRNN(dropout, spatial_dimensions, mdrnn_kw)]
        if num_spatial_dims == 2:
            convop = th.nn.Conv2d
            kernel = [1, 1]

        elif num_spatial_dims == 3:
            convop = th.nn.Conv3d
            kernel = [1, 1, 1]
        else:
            raise Exception('pytorch cannot handle more than 3 dimensions for convolution')
        if num_output is not None:
            model += [convop(num_hidden, num_output, kernel)]
            if not self.noactivation:
                model += [self.vwfc_activation()]
        self.model = th.nn.Sequential(*model)

    def forward(self, input):
        return self.model.forward(input)


generate_defaults_info(MDGRUBlock)