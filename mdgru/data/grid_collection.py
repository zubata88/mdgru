__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import copy
import logging
import os
from os import listdir
from os.path import isfile, isdir, join, splitext
from threading import Thread

import nibabel as nib
import mvloader.nifti as ni
import mvloader.nrrd as nr
import mvloader.dicom as dm
import pydicom
from mvloader.volume import Volume
import nrrd
import numpy as np
import skimage.io as skio
from scipy.misc import imsave, imread
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.measurements import label
from mdgru.helper import argget, counter_generator, compile_arguments, generate_defaults_info
from . import DataCollection


class GridDataCollection(DataCollection):
    w = [64, 64]
    pixdim = np.asarray([1, 1, 1, 1, 1, 1, 1])
    affine = np.eye(4)
    labellist = None
    _defaults = {
        'featurefiles': {'help': 'Filenames of featurefiles.', 'nargs': '+', 'short': 'f'},
        'maskfiles': {'value': [], 'help': 'Filenames of mask file(s) to be used as reference', 'short': 'm',
                      'nargs': '+'},
        'subtractGaussSigma': {'value': [5], 'type': int,
                               'help': 'Standard deviations to use for gaussian filtered image during highpass filtering data augmentation step. No arguments deactivates the feature. Can have 1 or nfeatures entries',
                               'nargs': '*'},
        'nooriginal': {'value': False, 'help': 'Do not use original data, only gauss filtered'},
        'correct_orientation': {'value': True, 'invert_meaning': 'dont_',
                                'help': 'Do not correct for the nifti orientation (for example, if header information cannot be trusted but all data arrays are correctly aligned'},
        'deform': {'value': [0], 'help': 'Deformation grid spacing in pixels. If zero, no deformation will be applied',
                   'type': int},
        'deformSigma': {'value': [0],
                        'help': 'Given a deformation grid spacing, this determines the standard deviations for each dimension of the random deformation vectors.',
                        'type': float},
        'mirror': {'value': [0], 'help': 'Activate random mirroring along the specified axes during training',
                   'type': bool},
        'gaussiannoise': {'value': False,
                          'help': 'Random multiplicative Gaussian noise on the input data with given std and mean 1'},
        'scaling': {'value': [0],
                    'help': 'Amount ot randomly scale images, per dimension, or for all dimensions, as a factor (e.g. 1.25)',
                    'type': float},
        'rotation': {'value': 0,
                     'help': 'Amount in radians to randomly rotate the input around a randomly drawn vector',
                     'type': float},
        'shift': {'value': [0],
                  'help': 'In order to sample outside of discrete coordinates, this can be set to 1 on the relevant axes',
                  'type': float},
        'vary_mean': 0,
        'vary_stddev': 0,
        'interpolate_always': {'value': False,
                               'help': 'Should we also interpolate when using no deformation grids (forces to use same pathways).'},
        'deformseed': {'value': 1234, 'help': 'defines the random seed used for the deformation variables',
                       'type': int},
        'interpolation_order': {'value': 3,
                                'help': 'Spline order interpolation. Values lower than 3 are: 0: nearest, 1: linear, 2: cubic.'},
        'padding_rule': {'value': 'constant',
                         'help': 'Rule on how to add values outside the image boundaries. options are: (‘constant’, ‘nearest’, ‘reflect’ or ‘wrap’'},
        'regression': False,
        'softlabels': False,
        'whiten': {'value': True, 'invert_meaning': 'dont_', 'help': 'Dont whiten data to mean 0 and std 1.'},
        'whiten_subvolumes': {'value': False,
                              'help': 'Whiten subvolumes to mean 0 and std 1 (usually it makes more sense to do so on whole volumes)'},
        'each_with_labels': {'value': 0, 'type': int, 'help': 'Force each n-th sample to contain labelled data'},
        'presize_for_normalization': {'value': [None],
                                      'help': 'Supply fixed sizes for the calculation of mean and stddev (only suitable with option whiten set)'},
        'half_gaussian_clip': False,
        'pyramid_sampling': False,
        'choose_mask_at_random': {'value': False,
                                  'help': 'if multiple masks are provided, we select one at random for each sample'},
        'zero_out_label': None,
        'lazy': {'value': True, 'help': 'Do not load values lazily', 'invert_meaning': 'non'},
        'perform_one_hot_encoding': {'value': True, 'help': 'Do not one hot encode target', 'invert_meaning': 'dont_'},
        'minlabel': {'value': 1, 'type': int, 'help': 'Minimum label to count for each_with_label functionality'},
        'channels_first': False,
        # {'value': True, 'help': 'Channels first or last? First is needed for nchw format (e.g. pytorch) and last is used by tensorflow'}
        'preloadall': False,
        'truncated_deform': {'value': False,
                             'help': 'deformations with displacements of maximum 3 times gausssigma in each spatial direction'},
        'connected_components': {'value': False,
                                 'help': 'return labels of connected components for each pixel belonging to a component instead of its label. Only works for binary segmentation and if no one hot encoding is used (with pytorch).'},
        'ignore_missing_mask': False,
    }

    def __init__(self, w, p, location=None, tps=None, kw={}):
        """

        Parameters
        ----------
        w : list
            subvolume/patchsize
        p : list
            amount of padding per dimension.
        location : str, optional
            Root folder where samples defined by featurefiles and maskfiles lie. Needs to be provided if tps is not.
        tps : list, optional
            List of locations or samples defined by featurefiles and maskfiles. Needs to be provided if location is not.

        """
        super(GridDataCollection, self).__init__(kw)
        self.origargs.update({"location": location, "tps": tps})
        self.w = np.ndarray.tolist(w) if not isinstance(w, list) else w
        self.p = p

        data_kw, kw = compile_arguments(GridDataCollection, kw, transitive=False)
        for k, v in data_kw.items():
            setattr(self, k, v)

        if tps is not None:
            self.tps = []
            [self.tps.extend(DataCollection.get_all_tps(t, self.featurefiles,
                                                        self.maskfiles if not self.ignore_missing_mask else [])) for t
             in tps]
        elif location is not None:
            if callable(location):
                self.tps = [location]
            else:
                self.tps = DataCollection.get_all_tps(location, self.featurefiles,
                                                      self.maskfiles if not self.ignore_missing_mask else [])
            if len(self.tps) == 0:
                raise Exception(
                    'no timepoints at location {} containing both {} and {}'.format(location, self.maskfiles,
                                                                                    self.featurefiles))
        else:
            raise Exception('either tps or location has to be set')

        if len(self.tps) == 0:
            raise Exception('there were no timepoints provided and location was not set')

        def oneorn(paramname):
            t = getattr(self, paramname)
            if len(t) == 1:
                setattr(self, paramname, t * len(self.w))
            elif len(t) == len(self.w) or len(t) == 0:
                return
            else:
                raise Exception('Parameter {} needs to have the same amount of entries as windowsize'.format(paramname))

        oneorn('p')
        oneorn('subtractGaussSigma')
        oneorn('deform')
        oneorn('deformSigma')
        oneorn('mirror')
        oneorn('scaling')
        oneorn('shift')
        oneorn('presize_for_normalization')

        if self.choose_mask_at_random:
            self.random_mask_state = np.random.RandomState(argget(kw, 'randommaskseed', 1337))
        self.imagedict = {}
        self.numoffeatures = argget(kw, 'numoffeatures', len(self._get_features_and_masks(self.tps[0])[0]))
        self.sample_counter = 0

    def load(self, file, lazy=True):
        """
        Handles all data loading from disk. If new filetypes should be allowed, this has to be implemented here.

        Parameters
        ----------
        file : str
            file path of the image / volume to load or folder of the images to load as volume.
        lazy : bool
            If set to False, all files are kept in memory once they are loaded.

        Returns
        -------
        image data
        """
        if not lazy:
            if file in self.imagedict.keys():
                return self.imagedict[file]
            else:
                self.imagedict[file] = self.load(file, True)
                self.imagedict[file] *= 1
                return self.imagedict[file]
        else:
            # individual files for each slice
            if isdir(file):
                # try loading each image and concatenating stuff.
                files = [f for f in sorted(listdir(file), key=str.lower) if isfile(join(file, f))]
                images = ['.png', '.pgm', '.pnm', '.tif', '.jpeg', '.jpg', '.tiff']
                if splitext(files[0])[-1].lower() in images:
                    # we assume we have natural image files which compose to a big volume
                    arr = []
                    for f in files:
                        fi = join(file, f)
                        if splitext(f)[-1].lower() in images:
                            arr.append(imread(fi))
                        else:
                            raise Exception(
                                'we implemented folderwise image volume reading only for the here listed types, not {}. feel free to contribute!'.format(
                                    splitext(f)))
                    return np.stack(arr)
                else:
                    # lets try to load this folder as a dicom folder. We assume, that all images related to the first
                    # image belong to the volume!
                    vol = dm.open_stack(file)
                    self.affine = vol.get_aligned_transformation("RAS")
                    f = vol.aligned_volume
                    return f

            else:
                # we got one file, nice!
                ending = splitext(file)[-1].lower()
                if ending in ['.nii', '.hdr', '.nii.gz', '.gz']:
                    if self.correct_orientation:
                        vol = ni.open_image(file, verbose=False)
                        self.affine = vol.get_aligned_transformation("RAS")
                        data = vol.aligned_volume
                    else:
                        f = nib.load(file)
                        self.affine = f.affine
                        self.pixdim = np.asarray(f.header['pixdim'][1:])
                        data = f.get_data()
                    return data
                elif ending in ['.nrrd', '.nhdr']:
                    if self.correct_orientation:
                        vol = nr.open_image(file, verbose=False)
                        self.affine = vol.get_aligned_transformation("RAS")
                        f = vol.aligned_volume
                    else:
                        try:
                            f, h = nrrd.read(file)
                        except:
                            print('could not read file {}'.format(file))
                            logging.getLogger('data').error('could not read file {}'.format(file))
                            raise Exception('could not read file {}'.format(file))
                        self.affine = np.eye(4)
                    return f
                elif ending in ['.dcm']:
                    f = pydicom.dcmread(file).pixel_array
                    return f
                elif ending in ['.mha']:
                    f = skio.imread(file, plugin='simpleitk')
                    self.affine = np.eye(4)
                    return f
                elif ending in ['.png', '.pgm', '.pnm']:
                    data = imread(file)
                    if len(data.shape) > 2:
                        return np.transpose(data, [2, 0, 1])
                    else:
                        return data
                    return imread(file)
                else:
                    raise Exception('{} not known'.format(ending))

    def save(self, data, filename, tporigin=None):
        """
        Saves image in data at location filename. Currently, data can only be saved as nifti or png images.

        Parameters
        ----------
        data : ndarray containing the image data
        filename : location to store the image
        tporigin : used, if the data needs to be stored in the same orientation as the data at tporigin. Only works for nifti files

        """
        try:
            ending = os.path.splitext(self.maskfiles[0])[-1]
        except Exception as e:
            ending = os.path.splitext(self.featurefiles[0])[-1]

        if ending in ['.nii', '.hdr', '.nii.gz', '.gz'] or len(data.squeeze().shape) > 2:
            if self.correct_orientation and tporigin is not None:
                # we corrected the orientation and we have the information to undo our wrongs, lets go:
                aligned_data = Volume(data, np.eye(4), "RAS")  # dummy initialisation if everything else fails
                try:
                    tporigin_vol = ni.open_image(os.path.join(tporigin, self.maskfiles[0]), verbose=False)
                except:
                    try:
                        tporigin_vol = ni.open_image(os.path.join(tporigin, self.featurefiles[0]), verbose=False)
                    except Exception as e:
                        logging.getLogger('data').warning('could not correct orientation for file {} from {}'
                                                          .format(filename, tporigin))
                        logging.getLogger('data').debug('because {}'.format(e))
                try:
                    aligned_vol = Volume(data, tporigin_vol.aligned_transformation, tporigin_vol.system)
                    aligned_data = aligned_vol.copy_like(tporigin_vol)
                except Exception as e:
                    logging.getLogger('data').warning('could not correct orientation for file {} from {}'
                                                      .format(filename, tporigin))
                    logging.getLogger('data').debug('because {}'.format(e))

                finally:
                    ni.save_volume(filename + ".nii.gz", aligned_data, True)
            else:
                if self.correct_orientation:
                    logging.getLogger('data').warning(
                        'could not correct orientation for file {} since tporigin is None: {}'
                            .format(filename, tporigin))
                nib.save(nib.Nifti1Image(data, self.affine), filename + ".nii.gz")
        else:
            if np.max(data) <= 1.0 and np.min(data) >= 0:
                np.int8(np.clip(data * 256, 0, 255))
            imsave(filename + ".png", data.squeeze())

    def preload_all(self):
        """
        Greedily loads all images into memory

        """
        for tp in self.tps:
            for f in self.featurefiles + self.maskfiles:
                file = os.path.join(tp, f)
                print('preloading {}'.format(file))
                self.load(file, lazy=False)

    def set_states(self, states):
        """
        Sets states of random generators according to the states in states

        Parameters
        ----------
        states : random generator states

        """
        if states is None:
            logging.getLogger('eval').warning(
                'could not reproduce state, setting unreproducable random seed for all random states')
            self.randomstate.seed(np.random.randint(0, 1000000))
            if hasattr(self, 'random_mask_state'):
                self.random_mask_state.seed(np.random.randint(0, 100000))
            if hasattr(self, 'deformrandomstate'):
                self.deformrandomstate.seed(np.random.randint(0, 100000))
        else:
            if hasattr(self, 'random_mask_state') and 'random_mask_state' in states:
                self.random_mask_state.set_state(states['random_mask_state'])
            if hasattr(self, 'deformrandomstate') and 'deformrandomstate' in states:
                self.deformrandomstate.set_state(states['deformrandomstate'])
            self.randomstate.set_state(states['randomstate'])

    def get_states(self):
        """
        Get the states of all involved random generators

        Returns
        -------
        states of random generators
        """
        states = {}
        if hasattr(self, 'random_mask_state'):
            states['random_mask_state'] = self.random_mask_state.get_state()
        if hasattr(self, 'deformrandomstate'):
            states['deformrandomstate'] = self.deformrandomstate.get_state()
        states['randomstate'] = self.randomstate.get_state()
        return states

    def get_shape(self):
        """
        Returns the shape of the input data (with the batchsize set to None)

        Returns
        -------
        list : shape of input data
        """
        if not self.channels_first:
            return [None] + self.w + [self.numoffeatures]
        else:
            return [None] + [self.numoffeatures] + self.w

    def get_target_shape(self):
        """
        Returns the shape of the target data

        Returns
        -------
        list : shape of target data
        """
        if not self.channels_first:
            return [None] + self.w + [self.nclasses]
        else:
            return [None] + [self.nclasses] + self.w

    def get_data_dims(self):
        """
        Returns the shape of all available data concatenated in the batch dimension

        Returns
        -------
        list : shape of all input data
        """
        return [len(self.tps)] + self.get_shape()[1:]

    def subtract_gauss(self, data):
        """
        Subtracts gaussian filtered data from itself

        Parameters
        ----------
        data : ndarray
            data to preprocess

        Returns
        -------
        ndarray : gaussian filtered data
        """
        return data - gaussian_filter(np.float32(data),
                                      np.asarray(self.subtractGaussSigma) * 1.0 / np.asarray(
                                          self.pixdim[:len(data.shape)]))

    def _get_features_and_masks(self, folder, featurefiles=None, maskfiles=None):
        """
        Returns for sample in folder all feature and mask files
        Parameters
        ----------
        folder : str
            location of sample
        featurefiles : list of str, optional
            featurefiles to return
        maskfiles : list of str, optional
            maskfiles to return
        Returns
        -------
            tuple of feature and mask ndarrays
        """
        if callable(folder):
            return folder()
        if featurefiles is None:
            featurefiles = self.featurefiles
        if maskfiles is None:
            maskfiles = self.maskfiles
        features = [self.load(os.path.join(folder, i), lazy=self.lazy).squeeze() for i in featurefiles]
        if len(self.subtractGaussSigma):
            if self.nooriginal:
                print("nooriginal")
                features = [self.subtract_gauss(f) for f in features]
            else:
                features.extend([self.subtract_gauss(f) for f in features])
        try:
            if self.choose_mask_at_random and len(maskfiles) > 1:  # chooses one of the masks at random
                m = self.random_mask_state.randint(0, len(maskfiles))
                maskfiles = [maskfiles[m]]
            masks = [self.load(os.path.join(folder, i), lazy=self.lazy) for i in maskfiles]
        except Exception:
            logging.getLogger('data').warning('Could not load mask files for sample {}'.format(folder))
            masks = []
        return features, masks

    def random_sample(self, batch_size=1, dtype=None, tp=None, **kw):
        """
        Randomly samples batch_size times from the data, using data augmentation if specified when creating the class

        Parameters
        ----------
        batch_size : number of samples to draw
        dtype : datatype to return
        tp : specific timepoint / patient to sample from
        kw: options (not used)

        Returns
        -------
        tuple of samples and corresponding label masks
        """
        batch = []
        labels = []
        for _ in range(batch_size):
            self.sample_counter += 1
            we_are_still_looking = True
            while (we_are_still_looking):
                we_are_still_looking = False
                if tp is not None:
                    self.curr_tps = tp
                else:
                    self.curr_tps = self.randomstate.randint(0, len(self.tps))
                folder = self.tps[self.curr_tps]
                try:
                    features, masks = self._get_features_and_masks(folder)
                except Exception as e:
                    print('could not load all data from {}, will now move to next random sample'.format(folder))
                    logging.getLogger('data').error(
                        'could not load all data from {}, will now move to next random sample'.format(folder))
                    return self.random_sample(batch_size, dtype, tp, **kw)
                shapev = [x for x in np.shape(features[0]) if x > 1]
                paddedw = np.ones(len(shapev))
                paddedw[:len(self.w)] = self.w
                valid_range = [x for x in shapev - paddedw]
                weonlyacceptwithlabels = self.each_with_labels > 0 and self.sample_counter % self.each_with_labels == 0
                if weonlyacceptwithlabels:
                    if np.sum(np.asarray(masks) >= self.minlabel) > 0:
                        # choose one of the labels, by picking a random index from lesion voxel 0 to nlesions-1
                        l = self.randomstate.randint(0, np.sum(np.asarray(masks) >= self.minlabel))
                        # now get the label coordinates
                        lab = [np.int64(ll[l]) for ll in np.where(np.squeeze(masks) >= self.minlabel)]
                        # now make sure, we look in a subwindowgroup of all possible windows, where this voxel is present
                        imin = [self.randomstate.randint(max(f - self.w[ind], 0), max(1, min(f + 1, valid_range[ind])))
                                for ind, f in enumerate(lab)]
                    else:
                        we_are_still_looking = True
                        continue
                else:
                    if self.pyramid_sampling:
                        r = self.randomstate.rand(len(valid_range))
                        rr = [(1 - (2 * rrr - 1) ** 2) / 2 if rrr < 0.5 else (1 + (2 * rrr - 1) ** 2) / 2 for rrr in r]
                        imin = [np.int32(rr[ind] * (max(j + self.p[ind], 1) + self.p[ind]) - self.p[ind]) for ind, j in
                                enumerate(valid_range)]
                    else:
                        imin = [self.randomstate.randint(0 - self.p[ind], max(j + self.p[ind], 1)) for ind, j in
                                enumerate(valid_range)]
                imax = [imin[i] + paddedw[i] for i in range(len(imin))]

                tempdata, templabels = self._extract_sample(features, masks, imin, imax, shapev,
                                                            needslabels=weonlyacceptwithlabels,
                                                            one_hot=self.perform_one_hot_encoding)
                if weonlyacceptwithlabels and len(templabels) == 0:
                    we_are_still_looking = True
                    continue
            batch.append(tempdata)
            labels.append(templabels)

        if dtype is not None:
            batch = np.asarray(batch, dtype=dtype)
        else:
            batch = np.asarray(batch)
        labels = np.asarray(labels)

        # if not self.perform_one_hot_encoding:
        #     order = [x for x in range(len(labels.shape))]
        #     order.pop(1)
        #     order.append(1)
        #     labels = np.transpose(labels, order)
        if self.channels_first:
            ndims = len(batch.shape)
            neworder = [0, ndims - 1] + [i for i in range(1, ndims - 1)]
            batch = np.transpose(batch, neworder)
            if self.perform_one_hot_encoding:
                labels = np.transpose(labels, neworder)
            elif self.connected_components:
                labels = label(labels)[0]  # only the labelled map, without number of components

        return batch, labels

    def transformAffine(self, coords):
        """
        Transforms coordinates according to the specified data augmentation scheme

        Parameters
        ----------
        coords : ndarray
            original, not augmented pixel coordinates of the subvolume / patch

        Returns
        -------
        "augmented" ndarray of coords
        """
        coordsshape = coords.shape
        dims = coordsshape[0] + 1
        coords = coords.reshape((len(coords), -1))
        coords = np.concatenate((coords, np.ones((1, len(coords[0])))), 0)
        affine = np.eye(dims)
        # now transform first to center:
        meanvec = np.mean(coords, 1)
        center = np.eye(dims)
        center[:-1, -1] = -meanvec[:-1]
        affine = np.matmul(center, affine)

        if np.sum(self.shift):
            affine[:-1, -1] += (self.deformrandomstate.rand(dims - 1) - 0.5) * np.float32(self.shift)
        if np.max(self.scaling) > 1:
            scales = np.ones(dims)
            # scales[:-1] = (self.deformrandomstate.rand(dims-1)-0.5)*(self.scaling-1.0/self.scaling)+(self.scaling+1/self.scaling)/2
            scales[:-1] = self.scaling ** (self.deformrandomstate.rand(dims - 1) * 2 - 1)
            scales = np.diag(scales)
            # print(scales)
            affine = np.matmul(scales, affine)
        if np.sum(self.rotation):
            affine = self._rotate(affine)
        # move back to location:
        center[:-1, -1] = -center[:-1, -1]
        affine = np.matmul(center, affine)
        # now appyl to coords:
        coords = np.matmul(affine, coords)
        coords = coords[:-1]
        coords = coords.reshape(coordsshape)
        return coords

    def _rotate(self, affine):
        """ Helper function to rotate an affine matrix"""
        dims = affine.shape[0]
        if not np.isscalar(self.rotation):
            raise Exception('this class requires exactly one entry for rotation!')
        theta = (self.deformrandomstate.rand() - 0.5) * 2 * self.rotation
        if dims == 4:

            # sample unit vector:
            u = np.random.random(3)
            u /= np.sqrt(np.sum([uu ** 2 for uu in u]) + 1e-8)
            ct = np.cos(theta)
            st = np.sin(theta)
            rot = np.eye(4)
            rot[:3, :3] = [
                [ct + u[0] ** 2 * (1 - ct), u[0] * u[1] * (1 - ct) - u[2] * st, u[0] * u[2] * (1 - ct) + u[2] * st],
                [u[1] * u[0] * (1 - ct) + u[2] * st, ct + u[1] ** 2 * (1 - ct), u[1] * u[2] * (1 - ct) - u[0] * st],
                [u[2] * u[0] * (1 - ct) - u[1] * st, u[2] * u[1] * (1 - ct) + u[0] * st, ct + u[2] ** 2 * (1 - ct)]]

        elif dims == 3:
            rot = np.eye(3)
            rot[:2, :2] = np.asarray([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        else:
            raise Exception(
                'implement this for each dimension, since not yet implemented for dimension {}'.format(dims))

        return np.matmul(rot, affine)

    def _extract_sample(self, features, masks, imin, imax, shapev, needslabels=False, one_hot=True):
        """
        Returns for one sample in the batch the extracted features and mask(s).
        the required output has shape [wx,wy,wz,f],[wx,wy,wz,c] with wxyz being
        the subvolumesize and f,c the features and classes respectively. Use
        onehot in here if onehot is used, optionally, if at all no one hot
        vector encoded data is used, the flag can be set to False

        Parameters
        ----------
        features : ndarray
            input data of full image
        masks : ndarray
            respective label maps for the full sample / patient / timepoint
        imin : list
            list of starting indices per dimension for the subvolume / patch to be extracted
        imax : list
            list of stopping indices per dimension for the subvolume / patch to be extracted
        shapev : list
            list defining the shape of features and mask
        needslabels : bool
            If set, will return an empty list if no labels are defined in the resulting subvolume / patch, forcing
            the calling method to call extract sample on a new location
        one_hot : bool
            Defines if we return the label data as one hot vectors per voxel / pixel or as label index per voxel / pixel

        Returns
        -------
        tuple of extracted data and labels corresponding to the patch / subvolume defined above and the chosen data
        augmentation scheme
        """

        # prepare containers
        tempdata = np.zeros([len(features)] + self.w, dtype=np.float32)
        featuredata = [f.squeeze() for f in features]
        templabels = []

        # accumulate mean and std for normalization
        if self.whiten and not self.whiten_subvolumes:
            numvoxs = [
                np.prod([s if g is None else g for g, s in zip(self.presize_for_normalization, f.squeeze().shape)]) for
                f in featuredata]
            means = [np.sum(f) * 1.0 / n for f, n in zip(featuredata, numvoxs)]
            stddevs = [np.sqrt(np.abs(np.mean((featuredata[i] - means[i]) ** 2))) for i in range(len(featuredata))]

        if np.sum(self.deform) + np.sum(self.rotation) + np.sum(self.scaling) + np.sum(
                self.shift) == 0 and not self.interpolate_always:  # No deformation/scaling/rotation
            # infer the valid part of subvolume in both source and target
            ranges = np.zeros((len(imin), 2), dtype=np.int32)
            ranges[:, 1] = 1
            ranges[:len(self.w), 1] = self.w
            imin = np.int32(imin)
            imax = np.int32(imax)
            for i in range(len(imin)):
                if imin[i] < 0:
                    ranges[i, 0] -= imin[i]
                    imin[i] -= imin[i]
                if imax[i] >= shapev[i]:
                    ranges[i, 1] -= ((imax[i] - shapev[i]))
                    imax[i] -= ((imax[i] - shapev[i]))
            # now index accordingly:
            targetindex = tuple([slice(None)] + [slice(np.int32(r[0]), np.int32(r[1])) for r in ranges])
            sourcesindex = tuple([slice(np.int32(mi), np.int32(ma)) for mi, ma in zip(imin, imax)])
            tempdata[targetindex] = np.asarray([f[sourcesindex] for f in featuredata])

            if len(masks):
                templabels = np.zeros(self.w, dtype=np.uint8)
                templabels[targetindex[1:]] = np.asarray([f.squeeze()[sourcesindex] for f in masks])
                if one_hot and not self.regression:
                    templabels = self._one_hot_vectorize(templabels, self.nclasses, zero_out_label=self.zero_out_label)


        else:  # we need to interpolate
            coords = np.float64(np.mgrid[[slice(np.int32(imi), np.int32(ima)) for imi, ima in zip(imin, imax)]])
            # coords = np.mgrid[imin[0]:imax[0],imin[1]:imax[1],imin[2]:imax[2]]
            coords = self.transformAffine(coords)
            if np.sum(self.deform):
                # create deformationfield:
                deform = self._get_deform_field_dm

                self.deformfield = deform()
                coords += self.deformfield

            # and set accordingly:
            if len(masks):
                if one_hot and not self.regression:
                    if len(masks) > 1:
                        logging.getLogger('data').error(
                            'cant have more than one mask with one_hot encoding in griddatacollection')
                    if self.softlabels:
                        mask = self._one_hot_vectorize(np.int32(masks[0]), self.nclasses,
                                                       zero_out_label=self.zero_out_label)
                        templabels = [map_coordinates(mask[..., c].squeeze(), coords, order=1, cval=np.float32(c == 0))
                                      for c in range(self.nclasses)]
                        templabels = np.concatenate([np.expand_dims(l, -1) for l in templabels], axis=-1)
                    else:
                        templabels = map_coordinates(masks[0].squeeze(), coords, order=0)
                        templabels = self._one_hot_vectorize(templabels, self.nclasses,
                                                             zero_out_label=self.zero_out_label)

                    if needslabels:
                        if np.sum(np.asarray(templabels[..., self.minlabel:])) == 0:
                            return [], []

                else:
                    # logging.getLogger('data').warning(
                    #     'maybe you want to revise this section before using! when do we not need a onehot?')
                    templabels = np.asarray(
                        [map_coordinates(f.squeeze(), coords, order=1 if self.softlabels else 0) for f in masks])
                    templabels = templabels.transpose([i for i in range(1, len(templabels.shape))] + [0])
                    if needslabels:
                        if np.sum(templabels >= self.minlabel) == 0:
                            return [], []
            tempdata = [map_coordinates(np.float32(f).squeeze(), coords, mode=self.padding_rule,
                                        order=self.interpolation_order) for f in features]
        tempdata = [x.reshape((self.w + [1])) for x in tempdata]  # FIXME: maybe we can just use expand_dims?
        if self.whiten:
            if self.whiten_subvolumes:
                raise Exception('not supported anymore')
                # for i in range(len(tempdata)):
                #     tempdata[i] = tempdata[i] - np.mean(tempdata[i])
                #     tempdata[i] /= np.sqrt(np.mean(tempdata[i] ** 2)) + 1e-20
            elif self.half_gaussian_clip:
                raise Exception('not supported anymore')
                # tempdata = [np.clip((x - means[i]) / (5 * stddevs[i]) - 1, -0.99999, 0.99999) for i, x in
                #             enumerate(tempdata)]
            else:
                tempdata = [(x - means[i]) / stddevs[i] for i, x in enumerate(tempdata)]
        if self.vary_mean > 0 or self.vary_stddev > 0:
            tempdata = [x * ((self.deformrandomstate.rand() - 0.5) * self.vary_stddev + 1) + (
                    self.deformrandomstate.rand() - 0.5) * self.vary_mean for x in tempdata]
        tempdata = np.concatenate(tempdata, -1)

        if np.sum(self.mirror):
            fr = []
            orig = []
            for i in self.mirror:
                fr.append(slice(None, None, np.int32(1 - self.deformrandomstate.randint(2) * i * 2)))
                orig.append(slice(None))
            fr.append(slice(None))  # features / labels
            orig.append(slice(None))
            tempdata[orig] = tempdata[fr]
            templabels[orig] = templabels[fr]
        if self.gaussiannoise > 0:
            tempdata *= (1 + (self.deformrandomstate.rand(*tempdata.shape) - 0.5) * self.gaussiannoise)
        return tempdata, templabels

    def get_volume_batch_generators(self):
        """
        Helper method returning a generator to efficiently fully sample a test volume on a predefined grid given w and p

        Returns
        -------
        Generator which completely covers the data for each sample in tps in a way defined by w and p
        """
        # volgeninfo = []
        def create_volgen(shape, w, padding, features, masks):
            w = np.asarray(w)
            padding = np.asarray(padding)
            W = w - padding * 2
            iters = np.int32(np.ceil((np.asarray([s for s in shape if s > 1]) + padding) * 1.0 / (W + padding)))
            for counts in counter_generator(iters):
                start = -padding + (w - padding) * counts
                end = (w - padding) * (counts + 1)
                subf, subm = self._extract_sample(features, masks, copy.deepcopy(start), copy.deepcopy(end), shape)
                ma = np.asarray([subm])
                fe = np.asarray([subf])
                if self.channels_first:
                    ndims = len(fe.shape)
                    neworder = [0, ndims - 1] + [i for i in range(1, ndims - 1)]
                    fe = np.transpose(fe, neworder)
                    ma = np.transpose(ma, neworder)
                yield fe, ma, start, end

        def volgeninfo(tps):
            for tp in tps:
                features, masks = self._get_features_and_masks(tp)
                spatial_shape = np.shape(features[0])
                volgen = create_volgen(spatial_shape, self.w, self.p, features, masks)
                yield [volgen, tp, spatial_shape, self.w, self.p]

        return volgeninfo(self.tps)

    def _get_deform_field_dm(self):
        """
        Helper function to get deformation field. First we define a low resolution deformation field, where we sample
        randomly from $N(0,I deformSigma)$ at each point in the grid. We then use cubic interpolation to upsample the
        deformation field to our resolution.

        Returns
        -------
        Deformation field which will be applied to the regular sampling coordinate ndarray
        """
        self.deformationStrength = self.deformrandomstate.rand()
        adr = [w // d + 4 for w, d in zip(self.w, self.deform)]
        deformshape = [len(self.w)] + adr
        tmp = np.zeros([4] * (len(self.w) - 1) + [len(self.w)] + self.w)

        if np.isscalar(self.deformSigma):
            myDeformSigma = np.array(len(self.w), self.deformSigma)
        else:
            myDeformSigma = np.asarray(self.deformSigma)

        strngs = [self.deformrandomstate.normal(0, myDeformSigma[i], deformshape[1:]) * self.deformationStrength
                  for i in range(len(myDeformSigma))]
        tdf = np.asarray(strngs, dtype=np.float32)

        if self.truncated_deform:
            upperBound = 3 * myDeformSigma
            for i in range(len(myDeformSigma)):
                overshoot_coordinates = np.where(np.abs(tdf[i]) > upperBound[i])
                while len(overshoot_coordinates[0]):
                    tdf[i][overshoot_coordinates] = np.float32(self.deformrandomstate.normal(0, myDeformSigma[i], len(
                        overshoot_coordinates[0])) * self.deformationStrength)
                    overshoot_coordinates = np.where(np.abs(tdf[i]) > upperBound[i])

        # logging.getLogger('data').info('truncated deformation field')

        def cint(x, pnm1, pn, pnp1, pnp2):
            return 0.5 * (
                    x * ((2 - x) * x - 1) * pnm1 + (x * x * (3 * x - 5) + 2) * pn + x * ((4 - 3 * x) * x + 1) * pnp1 + (
                    x - 1) * x * x * pnp2)

        r = [np.asarray([x * 1.0 / self.deform[i] - x // self.deform[i] for x in range(self.w[i])]).reshape(
            [self.w[i] if t == i + 1 else 1 for t in range(len(self.w) + 1)]) for i in range(len(self.w))]
        d = [np.asarray([x // self.deform[i] for x in range(self.w[i])]).reshape(
            [self.w[i] if t == i else 1 for t in range(len(self.w))]) for i in range(len(self.w))]

        if len(self.w) == 3:
            for i in range(4):
                for j in range(4):
                    xx = d[0] + i
                    yy = d[1] + j
                    zz = d[2] + 1
                    tmp[i, j] = cint(r[2], tdf[:, xx, yy, zz - 1], tdf[:, xx, yy, zz], tdf[:, xx, yy, zz + 1],
                                     tdf[:, xx, yy, zz + 2])
            for i in range(4):
                tmp[i, 0] = cint(r[1], tmp[i, 0], tmp[i, 1], tmp[i, 2], tmp[i, 3])
            return cint(r[0], tmp[0, 0], tmp[1, 0], tmp[2, 0], tmp[3, 0])

        elif len(self.w) == 2:
            for j in range(4):
                xx = d[0] + j
                yy = d[1] + 1
                tmp[j] = cint(r[1], tdf[:, xx, yy - 1], tdf[:, xx, yy], tdf[:, xx, yy + 1], tdf[:, xx, yy + 2])
            return cint(r[0], tmp[0], tmp[1], tmp[2], tmp[3])

        else:
            raise Exception('only implemented for 2d and 3d case. feel free to contribute')


class ThreadedGridDataCollection(GridDataCollection):

    _defaults = {'batch_size': 1,
                 'num_threads': {
                     'help': 'Determines how many threads are used to prefetch data, such that io operations do not cause delay.',
                     'value': 3, 'type': int},

                 }

    def __init__(self, featurefiles, maskfiles=[], location=None, tps=None, kw={}):
        """
        Threaded version of GridDataCollection. Basically a thin wrapper which employs num_threads threads to preload
        random samples. This will however result in possibly nonreproducible sampling patterns, as the threads run
        concurrently.

        Parameters
        ----------
        featurefiles : list of str
            filenames of the different features to consider
        maskfiles : list of str
            filenames of the available mask files per patient
        location : str, optional
            Location at which all samples containing all of featurefiles and maskfiles lie somewhere in the subfolder
            structure. Must be provided, if tps is not.
        tps : paths of all samples to consider. must be provided if location is not set.
        """
        super(ThreadedGridDataCollection, self).__init__(featurefiles, maskfiles, location, tps, kw)
        data_kw, kw = compile_arguments(ThreadedGridDataCollection, kw, transitive=False)
        for k, v in data_kw.items():
            setattr(self, k, v)

        # self.batch_size = argget(kw, 'batchsize', 1)
        self.curr_thread = 0
        self._batch = [None for _ in range(self.num_threads)]
        self._batchlabs = [None for _ in range(self.num_threads)]
        self._preloadthreads = [Thread(target=self._preload_random_sample, args=(self.batch_size, it,)) for it in
                                range(self.num_threads)]
        for t in self._preloadthreads:
            t.start()

    def random_sample(self, batch_size=1, dtype=None, tp=None, **kw):
        """
        Thin wrapper of GridDataCollections random sample, handling multiple threads to do the heavy lifting

        Parameters
        ----------
        batch_size : int
            batch_size. if this value is different to the one provided previously to the threads, data is discarded
            and new samples are computed adhering to the new batchsize.
        dtype :
            dtype of the returned input data
        tp : str
            specific timepoint to load
        kw : options, not used at the moment

        Returns
        -------
        Tuple of randomly sampled and possibly deformed / augmented input data and corresponding labels
        """
        if dtype is not None or tp is not None:
            logging.getLogger('data').warning(
                'cant handle any special terms in random sample in this step, will use next one. this will just return preloaded stuff. terms were: {},{},{},{}'.format(
                    batch_size, dtype, tp, ",".join(kw) + "(" + ",".join(kw.values()) + ")"))
        if self._preloadthreads[self.curr_thread] is not None:
            self._preloadthreads[self.curr_thread].join()
        if batch_size != self.batch_size:
            logging.getLogger('data').warning(
                'fetched wrong number of samples, need to fetch it now in order to get correct number of samples. updated batchsize accordingly')
            logging.getLogger('data').warning(
                'Did you forget to provide the threaded class with the correct batchsize at initialization?')
            self.batch_size = batch_size
            self._preloadthreads[self.curr_thread] = Thread(target=self._preload_random_sample,
                                                            args=(self.batch_size, self.curr_thread,))
            self._preloadthreads[self.curr_thread].start()
            self._preloadthreads[self.curr_thread].join()
        batch = np.copy(self._batch[self.curr_thread])
        batchlabs = np.copy(self._batchlabs[self.curr_thread])
        self._preloadthreads[self.curr_thread] = Thread(target=self._preload_random_sample,
                                                        args=(self.batch_size, self.curr_thread,))
        self._preloadthreads[self.curr_thread].start()
        self.curr_thread = (self.curr_thread + 1) % self.num_threads
        return batch, batchlabs

    def _preload_random_sample(self, batchsize, container_id):
        self._batch[container_id], self._batchlabs[container_id] = super(ThreadedGridDataCollection,
                                                                         self).random_sample(batch_size=batchsize)

generate_defaults_info(GridDataCollection)
generate_defaults_info(ThreadedGridDataCollection)
