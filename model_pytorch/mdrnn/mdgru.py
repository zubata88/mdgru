__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

from copy import copy, deepcopy

import torch as th

from helper import compile_arguments
from ..crnn.cgru import CGRUCell


class MDRNN(th.nn.Module):
    """MDRNN class originally designed to handle the sum of cGRU computations resulting in one MDGRU.

    _defaults contains initial values for most class attributes.
    :param use_dropconnect_x: Flag if dropconnect regularization should be applied to input weights
    :param use_dropconnect_h: Flag if dropconnect regularization should be applied to state weights
    :param swap_memory: Flag that trades slower computation with less memory consumption by swapping memory to CPU RAM
    :param return_cgru_results: Flag if instead of a sum, the individual cgru results should be returned
    :param use_static_rnn: Static rnn graph creation, not recommended
    :param no_avg_pool: Flag that defines if instead of average pooling convolutions with strides should be used
    :param filter_size_x: Dimensions of filters for the input (the current time dimension is ignored in each cRNN)
    :param filter_size_h: Dimensions of filters for the state (the current time dimension is ignored in each cRNN)
    :param crnn_activation: Activation function for the candidate / state / output in each cRNN
    :param legacy_cgru_addition: Activating old implementation of crnn sum, for backwards compatibility
    :param crnn_class: Which cRNN class should be used (CGRUCell for MDGRU)
    :param strides: Defines strides to be applied along each dimension

    :param inputarr: Input data, needs to be in shape [batch, spatialdim1...spatialdimn, channel]
    :param dropout: Dropoutrate to be applied
    :param dimensions: which dimensions should be processed with a cRNN (by default all of them)
    :param num_hidden: How many hidden units / channels does this MDRNN have
    :param name: What should be the name of this MDRNN

    """
    _defaults = {
        "use_dropconnect_x": {'value': True, 'help': "Should dropconnect be applied to the input?", 'invert':'dont_'},
        "use_dropconnect_h": {'value': True, 'help': "Should DropConnect be applied to the state?", 'invert': 'dont_'},
        # "swap_memory": True,
        "return_cgru_results": False,
        "use_static_rnn": False,
        "no_avg_pool": True,
        "filter_size_x": [7, 7, 7],
        "filter_size_h": [7, 7, 7],
        "crnn_activation": th.nn.Tanh,
        "legacy_cgru_addition": False,
        "crnn_class": CGRUCell,
        "strides": None,
        "name": "mdgru",
        "num_hidden": 100,
        "num_input": 6,
    }

    def __init__(self, dropout, spatial_dimensions, kw):
        super(MDRNN, self).__init__()
        mdgru_kw, kw = compile_arguments(self.__class__, kw, transitive=False)
        for k, v in mdgru_kw.items():
            setattr(self, k, v)
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
                # st = deepcopy(self.strides)
                # stontime = st.pop(d - 1)
                # if self.no_avgpool:
                #     def compress_time(data, fsize, stride, padding):
                #         fshape = fsize[1: -1] + [data.get_shape().as_list()[-1]] * 2
                #         filt = tf.get_variable("compressfilt", fshape)
                #         return convolution_helper_padding_same(data, filt, fshape, stride[1: -1])
                # else:
                #     if len(self.strides) == 3:
                #         compress_time = tf.nn.avg_pool3d
                #     elif len(self.strides) == 2:
                #         compress_time = tf.nn.avg_pool
                #     else:
                #         raise Exception("this wont work avg pool only implemented for 2 and 3d.")
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
