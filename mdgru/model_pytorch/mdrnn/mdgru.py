__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

from copy import copy, deepcopy

import torch as th

from mdgru.helper import compile_arguments, harmonize_filter_size, generate_defaults_info
from ..crnn.cgru import CGRUCell


class MDRNN(th.nn.Module):
    """MDRNN class originally designed to handle the sum of cGRU computations resulting in one MDGRU.

    _defaults contains initial values for most class attributes.
    :param dropout: Dropoutrate to be applied  (provided as keep rate)
    :param spatial_dimensions: which dimensions should be processed with a cRNN (by default all of them)
    """
    _defaults = {
        "use_dropconnect_x": {'value': True, 'help': "Should Dropconnect be applied to the input?", 'invert_meaning': 'dont_'},
        "use_dropconnect_h": {'value': True, 'help': "Should DropConnect be applied to the state?", 'invert_meaning': 'dont_'},
        # "swap_memory": True,
        "return_cgru_results": {'value': False, 'help': "Instead of summing, individual cgru channel results are concatenated."},
        "filter_size_x": {'value': [7], 'help': "Convolution kernel size for input."},
        "filter_size_h": {'value': [7], 'help': "Convolution kernel size for state."},
        "crnn_activation": {'value': th.nn.Tanh, 'help': "Activation function to be used for the CRNN."},
        "legacy_cgru_addition": {'value': False, 'help': "results in worse weight initialization, only use if you know what you are doing!"},
        "crnn_class": {'value': CGRUCell, 'help': 'CRNN class to be used in the MDRNN'}, #this is silly as we wont be able to ever display the correct help message if this is changed....
        "strides": None,
        "name": "mdgru",
        "num_hidden": 100,
        "num_input": 6,
    }

    def __init__(self, dropout, spatial_dimensions, kw):
        super(MDRNN, self).__init__()
        mdgru_kw, kw = compile_arguments(MDRNN, kw, transitive=False)
        for k, v in mdgru_kw.items():
            setattr(self, k, v)
        self.filter_size_x = harmonize_filter_size(self.filter_size_x, len(spatial_dimensions))
        self.filter_size_h = harmonize_filter_size(self.filter_size_h, len(spatial_dimensions))
        self.crnn_kw, kw = compile_arguments(self.crnn_class, kw, transitive=True)
        self.spatial_dimensions = spatial_dimensions
        self.dropout = dropout
        cgrus = []
        if self.use_dropconnect_h:
            self.crnn_kw["dropconnecth"] = self.dropout
        else:
            self.crnn_kw["dropconnecth"] = None
        if self.use_dropconnect_x:
            self.crnn_kw["dropconnectx"] = self.dropout
        else:
            self.crnn_kw["dropconnectx"] = None

        for d in self.spatial_dimensions:
            fsx = deepcopy(self.filter_size_x)
            fsh = deepcopy(self.filter_size_h)
            fsx.pop(d)
            fsh.pop(d)
            if self.strides is not None:
                raise Exception('we do not allow strides yet in the pytorch version')
            else:
                st = None

            crnn_dim_options = copy(self.crnn_kw)

            crnn_dim_options["filter_size_x"] = fsx
            crnn_dim_options["filter_size_h"] = fsh
            crnn_dim_options["strides"] = copy(st)

            # forward and back direction
            bicgru = th.nn.ModuleList([self.crnn_class(self.num_input, self.num_hidden, copy(crnn_dim_options)),
                                         self.crnn_class(self.num_input, self.num_hidden, copy(crnn_dim_options))])
            cgrus.append(bicgru)
        self.cgrus = th.nn.ModuleList(cgrus)

    def forward(self, input):
        outputs = []
        for i, (d, cgrus) in enumerate(zip(self.spatial_dimensions, self.cgrus)):
            # split tensor along d:
            cgru_split_input = th.unbind(input, d + 2) #spatial dim, hence d + 2
            output = th.stack(cgrus[0].forward(cgru_split_input), d + 2) \
                     + th.stack(cgrus[1].forward(cgru_split_input[::-1])[::-1], d + 2)
            outputs.append(output)
        # transform the sum to a mean over all cgrus (self.cgrus contains birnn)
        return th.sum(th.stack(outputs, dim=0), dim=0) / (len(self.cgrus) * 2)


generate_defaults_info(MDRNN)