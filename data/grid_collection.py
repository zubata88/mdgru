__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import copy
import logging
import os
from os import listdir
from os.path import isfile, isdir, join, splitext
from threading import Thread

import nibabel as nib
import nrrd
import numpy as np
import skimage.io as skio
from scipy.misc import imsave, imread
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.interpolation import zoom

from helper import argget, counter_generator
from helper import deprecated
from . import DataCollection, open_image, swap


class GridDataCollection(DataCollection):
    w = [64, 64]
    pixdim = np.asarray([1, 1, 1, 1, 1, 1, 1])
    affine = np.eye(4)
    labellist = None

    def __init__(self, featurefiles, maskfiles=[], location=None, tps=None, **kw):
        super(GridDataCollection, self).__init__(**kw)
        self.origargs.update({"featurefiles": featurefiles, "maskfiles": maskfiles, "location": location, "tps": tps})

        if not isinstance(featurefiles, list):
            featurefiles = [featurefiles]
        if not isinstance(maskfiles, list):
            maskfiles = [maskfiles]
        if tps is not None:
            self.tps = tps
        elif location is not None:
            if callable(location):
                self.tps = [location]
            else:
                self.tps = DataCollection.get_all_tps(location, featurefiles, maskfiles)
            if len(self.tps) == 0:
                raise Exception(
                    'no timepoints at location {} containing both {} and {}'.format(location, maskfiles, featurefiles))
        else:
            raise Exception('either tps or location has to be set')

        if len(self.tps) == 0:
            raise Exception('there were no timepoints provided and location was not set')

        self.featurefiles = featurefiles
        self.maskfiles = maskfiles
        w = argget(kw, 'w', self.w)
        if not isinstance(w, list):
            w = np.ndarray.tolist(w)
        self.w = w
        self.p = argget(kw, 'padding', np.zeros(np.shape(w)))

        self.deform = argget(kw, 'deformation', np.zeros(np.shape(w)))
        self.interpolate_always = argget(kw, 'interpolate_always', False)
        self.deformrandomstate = np.random.RandomState(argget(kw, 'deformseed', 1234))
        self.deformSigma = argget(kw, 'deformSigma', 5)
        if not np.isscalar(self.deformSigma) and len(self.deformSigma) != len(w):
            raise Exception(
                'we need the same sized deformsigma as w (hence, if we provide an array, it has to be the exact correct size)')
        self.deformpadding = 2
        self.datainterpolation = argget(kw, 'datainterpolation', 3)
        self.dataextrapolation = argget(kw, 'dataextrapolation', 'constant')
        self.dmdeform = argget(kw, 'deform_like_dm', True)

        self.scaling = np.float32(argget(kw, 'scaling', np.zeros(np.shape(w))))
        self.rotation = np.float32(argget(kw, 'rotation', 0))
        self.shift = np.float32(argget(kw, 'shift', np.zeros(np.shape(w))))
        self.mirror = np.float32(argget(kw, 'mirror', np.zeros(np.shape(w))))
        self.gaussiannoise = np.float32(argget(kw, 'gaussiannoise', 0.0))
        self.vary_mean = np.float32(argget(kw, 'vary_mean', 0))
        self.vary_stddev = np.float32(argget(kw, 'vary_stddev', 0))
        self.regression = argget(kw, 'regression', False)
        self.softlabels = argget(kw, 'softlabels', True)
        self.whiten = argget(kw, "whiten", True)
        self.each_with_labels = argget(kw, "each_with_labels", 0)
        if self.each_with_labels > 0 and len(self.maskfiles) == 0:
            raise Exception(
                'need to provide at leas tone mask file, otherwise we cant make sure we have labels set obviously')
        self.whiten_subvolumes = argget(kw, "whiten_subvolumes", False)
        self.presize_for_normalization = argget(kw, 'presize_for_normalization', [None for w in self.w])
        self.half_gaussian_clip = argget(kw, 'half_gaussian_clip', False)
        self.pyramid_sampling = argget(kw, 'pyramid_sampling', False)
        self.subtractGauss = argget(kw, "subtractGauss", False)
        self.nooriginal = argget(kw, 'nooriginal', False)
        self.subtractGaussSigma = np.float32(argget(kw, "sigma", 5))
        self.choose_mask_at_random = argget(kw, "choose_mask_at_random", False)
        if self.choose_mask_at_random:
            self.random_mask_state = np.random.RandomState(argget(kw, 'randommaskseed', 1337))
        self.zero_out_label = argget(kw, 'zero_out_label', None)
        self.running_mean = 0
        self.running_num = 0
        self.running_var = 0
        self.lazy = argget(kw, 'lazy', True)
        self.imagedict = {}
        self.perform_one_hot_encoding = argget(kw, 'perform_one_hot_encoding', True)
        self.correct_nifti_orientation = argget(kw, 'correct_nifti_orientation', len(self.w) == 3)
        if self.correct_nifti_orientation and len(self.w) != 3:
            self.correct_nifti_orientation = False
            logging.getLogger('data').warning('Can only correct for orientation for 3d data so far!')
        self.numoffeatures = argget(kw, 'numoffeatures', len(self._get_features_and_masks(self.tps[0])[0]))
        self.sample_counter = 0
        self.minlabel = argget(kw, 'minlabel', 1)

        if self.lazy == False and argget(kw, 'preloadall', False):
            self.preload_all()

    def load(self, file, lazy=True):

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
                arr = []
                for f in [f for f in sorted(listdir(file), key=str.lower) if isfile(join(file, f))]:
                    fi = join(file, f)
                    if splitext(f)[-1].lower() in ['.png', '.pgm', '.pnm', '.tif']:
                        arr.append(imread(fi))
                    else:
                        raise Exception(
                            'we implemented folderwise image volume reading only for the here listed types, not {}. feel free to contribute!'.format(
                                splitext(f)))
                return np.stack(arr)

            else:
                # we got one file, nice!
                ending = splitext(file)[-1].lower()
                if ending in ['.nii', '.hdr', '.nii.gz', '.gz']:
                    if self.correct_nifti_orientation:
                        vol = open_image(file, verbose=False)
                        self.affine = vol.get_aligned_matrix()
                        data = vol.aligned_volume
                    else:
                        f = nib.load(file)
                        self.affine = f.affine
                        self.pixdim = np.asarray(f.header['pixdim'][1:])
                        data = f.get_data()
                    return data
                elif ending in ['.nrrd', '.nhdr']:
                    try:
                        f, h = nrrd.read(file)
                    except:
                        print('could not read file {}'.format(file))
                        logging.getLogger('data').error('could not read file {}'.format(file))
                        raise Exception('could not read file {}'.format(file))
                    self.affine = np.eye(4)
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
        try:
            ending = os.path.splitext(self.maskfiles[0])[-1]
        except Exception as e:
            ending = os.path.splitext(self.featurefiles[0])[-1]

        if ending in ['.nii', '.hdr', '.nii.gz', '.gz'] or len(data.squeeze().shape) > 2:
            if self.correct_nifti_orientation and tporigin is not None:
                #we corrected the orientation and we have the information to undo our wrongs, lets go:
                dst_affine = np.eye(4)
                try:
                    dst_affine = nib.load(os.path.join(tporigin, self.maskfiles[0])).affine
                except:
                    try:
                        dst_affine = nib.load(os.path.join(tporigin, self.featurefiles[0])).affine
                    except Exception as e:
                        logging.getLogger('data').warning('could not correct orientation for file {} from {}'
                                                          .format(filename, tporigin))
                        logging.getLogger('data').debug('because {}'.format(e))
                try:
                    ndim = data.ndim

                    matrix = np.eye(ndim)
                    if len(matrix) > len(dst_affine):
                        #yes, elementwise. we only want to flip axes!!
                        matrix[:len(dst_affine), :len(dst_affine)] *= dst_affine
                    else:
                        matrix *= dst_affine[:len(matrix), :len(matrix)]
                    data = swap(data, matrix, ndim)
                except Exception as e:
                    logging.getLogger('data').warning('could not correct orientation for file {} from {} using {}'
                                                      .format(filename, tporigin, dst_affine))
                    logging.getLogger('data').debug('because {}'.format(e))

                finally:
                    nib.save(nib.Nifti1Image(data, dst_affine), filename + ".nii.gz")
            else:
                if self.correct_nifti_orientation:
                    logging.getLogger('data').warning('could not correct orientation for file {} since tporigin is None: {}'
                                                  .format(filename, tporigin))
                nib.save(nib.Nifti1Image(data, self.affine), filename + ".nii.gz")
        else:
            if np.max(data) <= 1.0 and np.min(data) >= 0:
                np.int8(np.clip(data * 256, 0, 255))
            imsave(filename + ".png", data.squeeze())

    def preload_all(self):
        for tp in self.tps:
            for f in self.featurefiles + self.maskfiles:
                file = os.path.join(tp, f)
                print('preloading {}'.format(file))
                self.load(file, lazy=False)

    def setStates(self, states):
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

    def getStates(self):
        states = {}
        if hasattr(self, 'random_mask_state'):
            states['random_mask_state'] = self.random_mask_state.get_state()
        if hasattr(self, 'deformrandomstate'):
            states['deformrandomstate'] = self.deformrandomstate.get_state()
        states['randomstate'] = self.randomstate.get_state()
        return states

    def get_shape(self):
        return [None] + self.w + [self.numoffeatures]

    def get_target_shape(self):
        return [None] + self.w + [None]

    def get_data_dims(self):
        return [len(self.tps)] + self.get_shape()[1:]

    def subtract_gauss(self, data):
        return data - gaussian_filter(np.float32(data),
                                      self.subtractGaussSigma * 1.0 / np.asarray(self.pixdim[:len(data.shape)]))

    def _get_features_and_masks(self, folder, featurefiles=None, maskfiles=None):
        if callable(folder):
            return folder()
        if featurefiles is None:
            featurefiles = self.featurefiles
        if maskfiles is None:
            maskfiles = self.maskfiles
        features = [self.load(os.path.join(folder, i), lazy=self.lazy).squeeze() for i in featurefiles]
        if self.subtractGauss:
            if self.nooriginal:
                print("nooriginal")
                features = [self.subtract_gauss(f) for f in features]
            else:
                features.extend([self.subtract_gauss(f) for f in features])
        if self.choose_mask_at_random and len(maskfiles) > 1:  # chooses one of the masks at random
            m = self.random_mask_state.randint(0, len(maskfiles))
            maskfiles = [maskfiles[m]]
        masks = [self.load(os.path.join(folder, i), lazy=self.lazy) for i in maskfiles]
        return features, masks

    def random_sample(self, batch_size=1, dtype=None, tp=None, **kw):
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

        if not self.perform_one_hot_encoding:
            order = [x for x in range(len(labels.shape))]
            order.pop(1)
            order.append(1)
            labels = np.transpose(labels, order)
        return batch, labels

    def transformAffine(self, coords):
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
        '''
        returns for one sample in the batch the extracted features and mask(s). 
        the required output has shape [wx,wy,wz,f],[wx,wy,wz,c] with wxyz being 
        the subvolumesize and f,c the features and classes respectively. Use 
        onehot in here if onehot is used, optionally, if at all no one hot 
        vector encoded data is used, the flag can be set to False
        '''

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
            targetindex = [slice(None)] + [slice(np.int32(r[0]), np.int32(r[1])) for r in ranges]
            sourcesindex = [slice(np.int32(mi), np.int32(ma)) for mi, ma in zip(imin, imax)]
            tempdata[targetindex] = np.asarray([f[sourcesindex] for f in featuredata])

            if len(masks):
                templabels = np.zeros(self.w, dtype=np.int8)
                templabels[targetindex[1:]] = np.asarray([f.squeeze()[sourcesindex] for f in masks])
                if one_hot and not self.regression:
                    templabels = self._one_hot_vectorize(templabels, self.nclasses, zero_out_label=self.zero_out_label)


        else:  # we need to interpolate
            coords = np.float64(np.mgrid[[slice(np.int32(imi), np.int32(ima)) for imi, ima in zip(imin, imax)]])
            # coords = np.mgrid[imin[0]:imax[0],imin[1]:imax[1],imin[2]:imax[2]]
            coords = self.transformAffine(coords)
            if np.sum(self.deform):
                # create deformationfield:
                if self.dmdeform:
                    deform = self._get_deform_field_dm
                else:
                    deform = self._get_deform_field

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
                    logging.getLogger('data').warning(
                        'maybe you want to revise this section before using! when do we not need a onehot?')
                    templabels = np.asarray(
                        [map_coordinates(f.squeeze(), coords, order=1 if self.softlabels else 0) for f in masks])
                    templabels = templabels.transpose([i for i in range(1, len(templabels.shape))] + [0])
                    if needslabels:
                        if np.sum(templabels >= self.minlabel) == 0:
                            return [], []
            tempdata = [map_coordinates(np.float32(f).squeeze(), coords, mode=self.dataextrapolation,
                                        order=self.datainterpolation) for f in features]
        tempdata = [x.reshape((self.w + [1])) for x in tempdata]  # FIXME: maybe we can just use expand_dims?
        if self.whiten:
            if self.whiten_subvolumes:
                raise Exception('not supported anymore')
                for i in range(len(tempdata)):
                    tempdata[i] = tempdata[i] - np.mean(tempdata[i])
                    tempdata[i] /= np.sqrt(np.mean(tempdata[i] ** 2)) + 1e-20
            elif self.half_gaussian_clip:
                raise Exception('not supported anymore')
                tempdata = [np.clip((x - means[i]) / (5 * stddevs[i]) - 1, -0.99999, 0.99999) for i, x in
                            enumerate(tempdata)]
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

    def sample_all(self, data_set=None, batch_size=1, **kw):
        raise Exception('implement this')

    def get_volume_batch_generators(self):
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
                yield np.asarray([subf]), ma, start, end

        def volgeninfo(tps):
            for tp in tps:
                features, masks = self._get_features_and_masks(tp)
                shape = np.shape(features[0])
                volgen = create_volgen(shape, self.w, self.p, features, masks)
                yield [volgen, tp, shape, self.w, self.p]

        return volgeninfo(self.tps)

    def _get_deform_field(self):
        self.deformationStrength = self.deformrandomstate.rand()
        deformshape = [3] + [(w - 1) // d + 2 + self.deformpadding for w, d in zip(self.w, self.deform)]
        df = np.float32(self.deformrandomstate.normal(0, self.deformSigma,
                                                      deformshape) * self.deformationStrength)  # we need 2 at least
        df = zoom(df, [1] + [ww / (1.0 * (d - self.deformpadding)) for ww, d in zip(self.w, deformshape[1:])], order=2)
        # center df from padding:
        if self.deformpadding:
            if len(self.w) == 2:
                df = df[:,
                     (df.shape[1] - self.w[0]) // 2:(df.shape[1] - self.w[0]) // 2 + self.w[0],
                     (df.shape[2] - self.w[1]) // 2:(df.shape[2] - self.w[1]) // 2 + self.w[1],
                     ]
            elif len(self.w) == 3:
                df = df[:,
                     (df.shape[1] - self.w[0]) // 2:(df.shape[1] - self.w[0]) // 2 + self.w[0],
                     (df.shape[2] - self.w[1]) // 2:(df.shape[2] - self.w[1]) // 2 + self.w[1],
                     (df.shape[3] - self.w[2]) // 2:(df.shape[3] - self.w[2]) // 2 + self.w[2],
                     ]
            else:
                raise Exception('this is only implemented for 2 and 3d case')

        return df

    def _get_deform_field_dm(self):
        self.deformationStrength = self.deformrandomstate.rand()
        adr = [w // d + 4 for w, d in zip(self.w, self.deform)]
        deformshape = [len(self.w)] + adr
        tmp = np.zeros([4] * (len(self.w) - 1) + [len(self.w)] + self.w)
        if np.isscalar(self.deformSigma):
            tdf = np.float32(self.deformrandomstate.normal(0, self.deformSigma,
                                                           deformshape) * self.deformationStrength)  # we need 2 at least
        else:
            strngs = [self.deformrandomstate.normal(0, self.deformSigma[i], deformshape[1:]) * self.deformationStrength
                      for i in range(len(self.deformSigma))]
            strngs = np.asarray(strngs)
            tdf = np.float32(np.asarray(strngs))

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

    @classmethod
    def createDataCollections(cls, featurefiles, maskfiles, location=None, tps=None, training=0.8, testing=0.2,
                              validation=0, **kw):
        if not isinstance(featurefiles, list):
            featurefiles = [featurefiles]
        if not isinstance(maskfiles, list):
            maskfiles = [maskfiles]
        if (validation + training + testing != 1):
            raise Exception('the proportions need to sum up to one!')
        if tps is not None:
            pass
        elif location is not None:
            tps = DataCollection.get_all_tps(location, featurefiles, maskfiles)
        else:
            raise Exception('either tps or location has to be set')
        np.random.seed(argget(kw, "seed", 12345678))
        np.random.shuffle(tps)
        trnum = np.int32(len(tps) * training)
        tenum = np.int32(len(tps) * testing)

        teset = cls(featurefiles, maskfiles, tps=tps[:tenum], **kw)
        if validation == 0 and trnum + tenum != len(tps):
            trnum += len(tps) - trnum - tenum  # add lost tp to tr
            valset = teset
        else:
            valset = cls(featurefiles, maskfiles, tps=tps[trnum + tenum:], **kw)
        trset = cls(featurefiles, maskfiles, tps=tps[tenum:trnum], **kw)

        return {'train': trset, 'validation': valset, 'test': teset}


class ThreadedGridDataCollection(GridDataCollection):
    def __init__(self, featurefiles, maskfiles=[], location=None, tps=None, **kw):
        super(ThreadedGridDataCollection, self).__init__(featurefiles, maskfiles, location, tps, **kw)

        self._batchsize = argget(kw, 'batchsize', 1)
        self.num_threads = argget(kw, 'num_threads', 1)
        self.curr_thread = 0
        self._batch = [None for _ in range(self.num_threads)]
        self._batchlabs = [None for _ in range(self.num_threads)]
        self._preloadthreads = [Thread(target=self._preload_random_sample, args=(self._batchsize, it,)) for it in
                                range(self.num_threads)]
        for t in self._preloadthreads:
            t.start()

    def random_sample(self, batch_size=1, dtype=None, tp=None, **kw):
        if dtype is not None or tp is not None:
            logging.getLogger('data').warning(
                'cant handle any special terms in random sample in this step, will use next one. this will just return preloaded stuff. terms were: {},{},{},{}'.format(
                    batch_size, dtype, tp, ",".join(kw) + "(" + ",".join(kw.values()) + ")"))
        if self._preloadthreads[self.curr_thread] is not None:
            self._preloadthreads[self.curr_thread].join()
        if batch_size != self._batchsize:
            logging.getLogger('data').warning(
                'fetched wrong number of samples, need to fetch it now in order to get correct number of samples. updated batchsize accordingly')
            logging.getLogger('data').warning(
                'Did you forget to provide the threaded class with the correct batchsize at initialization?')
            self._batchsize = batch_size
            self._preloadthreads[self.curr_thread] = Thread(target=self._preload_random_sample,
                                                            args=(self._batchsize, self.curr_thread,))
            self._preloadthreads[self.curr_thread].start()
            self._preloadthreads[self.curr_thread].join()
        batch = np.copy(self._batch[self.curr_thread])
        batchlabs = np.copy(self._batchlabs[self.curr_thread])
        self._preloadthreads[self.curr_thread] = Thread(target=self._preload_random_sample,
                                                        args=(self._batchsize, self.curr_thread,))
        self._preloadthreads[self.curr_thread].start()
        self.curr_thread = (self.curr_thread + 1) % self.num_threads
        return batch, batchlabs

    def _preload_random_sample(self, batchsize, container_id):
        self._batch[container_id], self._batchlabs[container_id] = super(ThreadedGridDataCollection,
                                                                         self).random_sample(batch_size=batchsize)

