__author__ = "Simon Andermatt, Simon Pezold"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import copy
import logging
import os
import subprocess
import sys
import numpy as np
from mdgru.helper import argget, compile_arguments, generate_defaults_info


class DataCollection(object):
    '''Abstract class for all data handling classes. 

    '''
    _defaults = {'seed': {'help': 'Seed to be used for deterministic random sampling, given no threading is used', 'value': 1234},
                 'nclasses': None,
                 }

    def __init__(self, kw):
        self.origargs = copy.copy(kw)
        data_kw, kw = compile_arguments(DataCollection, kw, transitive=False)
        for k, v in data_kw.items():
            setattr(self, k, v)
        self.randomstate = np.random.RandomState(self.seed)

    def set_states(self, state):
        ''' reset random state generators given the states in "states"

        Parameters
        ----------
        states: object
            Random generator state
        '''
        if state is None:
            logging.getLogger('eval').warning('could not reproduce state, setting unreproducable random seed')
            self.randomstate.set_seed(np.random.randint(0, 1000000))
        self.randomstate.set_state(state)

    def get_states(self):
        ''' Get states of this data collection'''
        return self.randomstate.get_state()

    def reset_seed(self, seed=12345678):
        ''' reset main random number generator with given seed '''
        self.randomstate = np.random.RandomState(seed)

    def random_sample(self, **kw):
        '''Randomly samples from our dataset. If the implementation knows 
        different datasets, the dataset string can be used to choose one, if 
        not, it will be ignored. 
        
        Parameters
        ----------
        \*\*kw: keyword args
            batch_size can be set, amongst other parameters. See implementing methods for
            more detail.
                  
        Returns
        -------
        array
            A random sample of length batch_size.
        
        '''
        raise Exception("random_sample not implemented in {}"
                        .format(self.__class__))

    def get_shape(self):
        raise Exception("needs to be implemented. should return batch shape" +
                        "with batch size set to None")

    def get_data_dims(self):
        '''Returns the dimensionality of the whole collection (even if samples 
        are returned/computed on the fly, the theoretical size is returned).
        Has between two and three entries (Depending on the type of data. A 
        dataset with sequence of vectors has 3, a dataset with sequences of 
        indices has two, etc)
        
        Returns
        -------
        list
            A shape array of the dimensionality of the data.
            
        '''
        raise Exception("get_data_dims not implemented in {}"
                        .format(self.__class__))

    def _one_hot_vectorize(self, indexlabels, nclasses=None, zero_out_label=None):
        '''
        simplified onehotlabels method. we discourage using interpolated labels 
        anyways, hence this only allows integer values in indexlabels

        Parameters
        ----------
        indexlabels : ndarray
            array containing labels or indices for each class, starting at 0 until nclasses-1
        nclasses : int
            number of classes
        zero_out_label : int
            label to assign probability of zero for the whole probability distribution

        Returns
        -------
        ndarray
            Probabilitydistributions per pixel where at position indexlabels the value is set to 1, otherwise to 0

        '''
        if nclasses is None:
            nclasses = self.nclasses
        # we reshape it into dims*classes, onehotvectorize it, and shape it back:
        lshape = indexlabels.shape
        lsprod = np.prod(lshape)
        l = np.zeros([lsprod, nclasses], dtype=np.int32)
        indexlabels = indexlabels.flatten()
        # print(l.shape)
        # print(np.max(indexlabels))
        # print(lshape)
        l[np.arange(0, lsprod, dtype=np.int32), indexlabels] = 1
        if zero_out_label is not None:
            l[:, zero_out_label] = 0
        # go back to shape from before.
        l = l.reshape(list(lshape) + [nclasses])
        return l

    @staticmethod
    def get_all_tps(folder, featurefiles, maskfiles):
        '''
        computes list of all folders that are subfolders of folder and contain all provided featurefiles and maskfiles.

        Parameters
        ----------
        folder: str
            location at which timepoints are searched
        featurefiles: list of str
            necessary featurefiles to be contained in a timepoint
        maskfiles: list of str
            necessary maskfiles to be contained in a timepoint

        Returns
        -------
        sorted list
            valid timepoints in string format
        '''
        comm = "find '" + os.path.join(folder, '') + "' -type d -exec test -e {}/" + featurefiles[0]
        for i in featurefiles[1:]:
            comm += " -a -e {}/" + i
        for i in maskfiles:
            comm += " -a -e {}/" + i
        comm += " \\; -print\n"
        res, err = subprocess.Popen(comm, stdout=subprocess.PIPE, shell=True).communicate()
        # print(comm)
        if (sys.version_info > (3, 0)):
            # Python 3 code in this block
            return sorted([str(r, 'utf-8') for r in res.split() if r])
        else:
            # Python 2 code in this block
            return sorted([str(r) for r in res.split() if r])


generate_defaults_info(DataCollection)