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
        "use_dropconnect_x": True,
        "use_dropconnect_h": True,
        "swap_memory": True,
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

    def forward(self, input):

        # outputs = []

        for i, (d, cgrus) in enumerate(zip(self.spatial_dimensions, self.cgrus)):
            # split tensor along d:
            cgru_split_input = th.unbind(input, d + 2) #spatial dim, hence d + 2
            output = th.stack(cgrus[0].forward(cgru_split_input), d + 2) \
                     + th.stack(cgrus[1].forward(cgru_split_input[::-1])[::-1], d + 2)
            if i == 0:
                outputs = output
            else:
                outputs += output
        outputs /= len(self.cgrus) * 2 # transform the sum to a mean over all cgrus (self.cgrus contains birnn)
        return outputs

    def __init__(self, dropout, spatial_dimensions, kw):
        super(MDRNN, self).__init__()
        mdgru_kw, kw = compile_arguments(self.__class__, kw, transitive=False)
        for k, v in mdgru_kw.items():
            setattr(self, k, v)
        self.crnn_kw, kw = compile_arguments(self.crnn_class, kw, transitive=True)
        self.spatial_dimensions = spatial_dimensions
        self.dropout = dropout
        self.cgrus = th.nn.ModuleList([])
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
            self.cgrus += [th.nn.ModuleList([self.crnn_class(self.num_input, self.num_hidden, copy(crnn_dim_options)),
                                         self.crnn_class(self.num_input, self.num_hidden, copy(crnn_dim_options))])]

    #         dimorder = [i for i in range(len(self.inputarr.get_shape()))]
    #         dimorder.insert(-1, dimorder.pop(d))
    #         dimorder = np.asarray(dimorder)
    #         # transpose to correct run direction
    #         trans_input = tf.transpose(self.inputarr, dimorder)
    #         myshape = [int(x) for x in trans_input.get_shape()[1:]]
    #         myshape.insert(0, -1)
    #         input_channels = int(trans_input.get_shape()[-1])
    #         current_time = int(myshape[-2])
    #         tempshape = [-1, current_time, input_channels]
    #         # transpose both back:
    #         dimorderback = [i for i in range(len(self.inputarr.get_shape()))]
    #         dimorderback.insert(d, dimorderback.pop(-2))
    #
    #         # forward direction:
    #         with tf.variable_scope("forward"):
    #             trans_resf = self.add_cgru(trans_input, myshape, tempshape, fsx=fsx, fsh=fsh, strides=copy(st))
    #             if self.strides is not None:
    #                 ksize = [1 for _ in self.inputarr.get_shape()]
    #                 ksize[-2] = stontime
    #                 trans_resf = compress_time(trans_resf,
    #                                            [int(k) if k >= 1 else int(np.round(1 // k)) for k in ksize],
    #                                            ksize, "SAME")
    #             resf = tf.transpose(trans_resf, dimorderback)
    #             outputs.append(resf)
    #         # backward direction
    #         with tf.variable_scope("backward"):
    #             rev_trans_input = tf.reverse(trans_input, axis=[len(dimorder) - 2])
    #             # dimorder should now be 1 at location d;)
    #             rev_trans_resb = self.add_cgru(rev_trans_input, myshape, tempshape, fsx=fsx, fsh=fsh,
    #                                            strides=copy(st))
    #             if self.strides is not None:
    #                 ksize = [1 for _ in self.inputarr.get_shape()]
    #                 ksize[-2] = stontime
    #                 rev_trans_resb = compress_time(rev_trans_resb,
    #                                                [int(k) if k >= 1 else int(np.round(1 // k)) for k in ksize],
    #                                                ksize, "SAME")
    #             trans_resb = tf.reverse(rev_trans_resb, axis=[len(dimorder) - 2])
    #             resb = tf.transpose(trans_resb, dimorderback)
    #         outputs.append(resb)
    #
    #     if self.return_cgru_results:
    #         return tf.concat(outputs, len(self.inputarr.get_shape()) - 1)
    #     else:
    #         if self.legacy_cgru_addition:
    #             return tf.add_n(outputs)
    #         else:
    #             return tf.add_n(outputs) / len(outputs)
    #
    # def add_cgru(self, minput, myshape, tempshape, fsx=[7, 7], fsh=[7, 7], strides=None):
    #     """Convenience method to unify the cRNN computation, gets input and shape and returns the cRNNs results."""
    #     kw = copy(self.crnn_kw)
    #     if self.use_dropconnect_h:
    #         kw["dropconnecth"] = self.dropout
    #     else:
    #         kw["dropconnecth"] = None
    #     if self.use_dropconnect_x:
    #         kw["dropconnectx"] = self.dropout
    #     else:
    #         kw["dropconnectx"] = None
    #     kw["filter_size_x"] = fsx
    #     kw["filter_size_h"] = fsh
    #     kw["strides"] = strides
    #     mycell = self.crnn_class(myshape, self.num_hidden, kw)
    #     trans_input_flattened = tf.reshape(minput, shape=tempshape)
    #     output_shape = deepcopy(myshape)
    #     if strides is not None:
    #         if len(strides) != len(output_shape) - 3:
    #             raise Exception("strides should match the current spatial dimensions")
    #         output_shape[1: -2] = [int(np.ceil((myshape[1 + i]) / strides[i])) for i in range(len(strides))]
    #     output_shape[-1] = mycell.output_size
    #
    #     zeros_dims = tf.stack([tf.shape(minput)[0], np.prod(output_shape[1: -2]), mycell.output_size])
    #     initial_state = tf.reshape(tf.fill(zeros_dims, 0.0), [-1, mycell.output_size])
    #     if self.use_static_rnn:
    #         trans_res_flattened, _ = tf.contrib.rnn.static_rnn(mycell, tf.unstack(trans_input_flattened, axis=-2),
    #                                                            dtype=tf.float32, initial_state=initial_state)
    #     else:
    #         trans_res_flattened, _ = tf.nn.dynamic_rnn(mycell, trans_input_flattened, dtype=tf.float32,
    #                                                    swap_memory=self.swap_memory, initial_state=initial_state)
    #
    #     return tf.reshape(trans_res_flattened, shape=output_shape)
