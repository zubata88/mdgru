__author__ = "Simon Andermatt, Simon Pezold"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

import copy
import logging
import os
import subprocess
import sys
import nibabel as nib
import numpy as np
from nibabel import quaternions
from helper import argget


def pos(system):
    """
    Return a tuple (rl, ap, si) where <rl> holds the index of "R" or "L", <ap>
    holds the index of "A" or "P", and <si> holds the index of "S" or "I" in
    the <system> string (case-insensitive).
    """
    return (index(system, "R"), index(system, "A"), index(system, "S"))


def index(system, character):
    """
    Get the index that the given <character> (or its opposite) has in the given
    <system> (strings expected; case-insensitive).
    """
    system = system.upper()
    character = character.upper()

    index = system.find(character)
    index = system.index(opposites()[character]) if index == -1 else index
    # ^ str.find() returns "-1" for mismatch, while str.index() raises an error

    return index


def opposites():
    """
    Return a dictionary that for every letter defining an anatomical direction
    and given as a key, will return the opposite direction in the same axis
    (using upper-case letters).

    As an example, opposites()["R"] will give "L".
    """
    return {"R": "L", "A": "P", "S": "I",
            "L": "R", "P": "A", "I": "S"}


def matrix(src, dst):
    """
    Calculate the rotation/reflection matrix that maps from the given <src> 3D
    anatomical coordinate system to the given <dst> system, as well as its
    inverse.

    Both systems are expected as a three-character string, where each character
    defines the positive direction of one of the anatomical axes. Valid
    characters are "R"/"L" for right/left, "A"/"P" for anterior/posterior, and
    "S"/"I" for superior/inferior positive axis direction (case-insensitive).

    Return a tuple (src2dst, dst2src) where <src2dst> holds the rotation and
    reflection matrix that maps coordinates from the <src> system into the
    <dst> system, and <dst2src> holds the matrix for the inverse mapping.
    Both matrices are (3, 3)-shaped Numpy arrays.
    """
    src = src.upper()  # For case-insensitivity
    dst = dst.upper()

    # Find the "R/L", "A/P", "S/I" positions
    src_pos = pos(src)
    dst_pos = pos(dst)

    # Build the transformation matrix:
    ndim = 3
    m = np.zeros((ndim, ndim), dtype=np.int)
    for i in range(ndim):
        # (-1) ** (...): If the character for the current axis is not the same
        # in the source and destination string, this means we have to mirror
        # the respective axis. The exponent then evaluates to True (i.e. 1),
        # and the power becomes -1. If the character is the same, the exponent
        # evaluates to False (i.e. 0), and the power becomes 1.
        m[dst_pos[i], src_pos[i]] = (-1) ** (dst[dst_pos[i]] != src[src_pos[i]])

    # Build the inverse transformation matrix:
    m_inv = np.asarray(np.round(np.linalg.inv(m)), dtype=np.int)
    return (m, m_inv)


def validate_swap_matrix(matrix, ndim):
    """
    Validate a rotation/reflection <matrix> that maps the axes of different
    <ndim>-dimensional coordinate systems. A matrix is considered valid if

        (1) abs(det(matrix[:ndim, :ndim])) == 1 and
        (2) sum(abs(matrix[:ndim, :ndim])) == <ndim>.

    Raise an error if the matrix is invalid, otherwise simply return.
    """
    msg = ""
    if np.abs(np.linalg.det(matrix[:ndim, :ndim])) != 1:
        msg = "abs(det(matrix[:ndim, :ndim])) != 1"
    elif np.sum(np.abs(matrix[:ndim, :ndim])) != ndim:
        msg = "sum(abs(matrix[:ndim, :ndim])) != %i" % ndim
    if msg:
        raise ValueError("The given matrix is not valid (%s)" % msg)


def swap(volume, matrix, ndim):
    """
    Swap the values in the given <volume> (<ndim>-dimensional Numpy array
    expected) according to the given <matrix> ((<ndim>, <ndim>)-shaped Numpy
    array expected, if a bigger one is given, the upper left <ndim>x<ndim> area
    is considered).

    The given <matrix> should represent a rotation/reflection matrix that maps
    the coordinate axes of the source coordinate system exactly onto the axes
    of the destination coordinate system (see the <validate_swap_matrix()>
    function).

    Return the swapping result as a new Numpy array.
    """

    matrix = matrix[:ndim, :ndim]

    validate_swap_matrix(matrix, ndim)
    if volume.ndim != ndim:
        raise ValueError("The given volume is not valid (ndim != %i)" % ndim)

    # New instance for manipulation
    volume = np.array(volume)

    # Invert the axes as necessary: Sum the columns of the transformation
    # matrix. Get a three-tuple, where each element is either +1, meaning the
    # respective axis (in source coordinates) doesn't have to be inverted, or
    # -1, meaning it has to be inverted.
    inv = np.round(np.sum(matrix, axis=0)).astype(np.int)
    # ^ Cast to int here to avoid deprecation warning in slice creation

    # Do the actual inversion
    volume = volume[tuple([slice(None, None, inv[n]) for n in range(ndim)])]
    #                      ^ represents either "::-1" (invert) or "::1" (don't)

    # Swap the axes as necessary: transform a vector representing the axis
    # numbers (i.e. (0, 1, 2)) by the absolute value of the given matrix (as
    # the inversions do not matter here), then permute the axes according to
    # the result
    perms = np.round(np.dot(abs(matrix), range(ndim))).astype(np.int)
    volume = np.transpose(volume, perms)

    return volume


class Volume(object):
    """
    A class that represents 3D scan volumes in a desired anatomical coordinate
    system (default is "RAS"). It is meant to serve as a layer on top of
    specific image formats (with different coordinate system conventions).

    Specifically, it is meant to make the spatial transformations of the
    underlying data somewhat transparent: it provides access to the voxel data
    via the field <aligned_volume>, which, if the desired coordinate system is
    "RAS", holds an array where axis 0 is guaranteed to point into the voxel
    coordinate axis of positive R (i.e. "right"), axis 1 maps to positive A
    (i.e. "anterior"), and axis 2 maps to positive S (i.e. "superior")
    (respective mappings apply for any other desired coordinate system). If an
    exact alignment is not possible without interpolation (because the scan
    directions were not exactly aligned with the coordinate axes), the array
    axes point into the directions of the original data that are closest to the
    respective coordinate axes.
    """

    def __init__(self, voxel_data, img=None, matrix=None, quaternion=None,
                 src_system=None, dst_system="RAS"):
        """
        Create a new instance. Provide the source voxel data via <voxel_data>
        (three-dimensional Numpy array expected, later available via the
        <src_volume> attribute), provide the respective source anatomical
        coordinate system via <system> (string expected, e.g. "LPS", see the
        <transformations.anatomical_coords> module). Optionally, specify a
        coordinate system in which the coordinates shall later be represented
        (defaults to "RAS"). Optionally provide the object returned by the
        respective file loader as <img> (may be useful for debugging).

        Additionally, provide either (1) a transformation <matrix> ((4, 4)-
        element Numpy array expected) that maps the source voxel indices to
        the source coordinate system's coordinates or (2) a three-tuple (q,
        s, o) via <quat> that does the same job: in the latter case, <q> is
        expected to be a four-tuple representing a unit quaternion (which
        itself represents a rotation), <s> is expected to be a three-tuple
        giving the voxel sizes in the source coordinate system, and <o> is
        expected to be a three-tuple representing an offset in the source
        coordinate system that is applied after the quaternion rotation and
        adjustment for voxel size.
        """
        if matrix is None and quaternion is None:
            raise ValueError("Provide either a matrix or quaternion " +
                             "information (neither is given)")
        if not (matrix is None) and not (quaternion is None):
            raise ValueError("Provide either a matrix or quaternion " +
                             "information (both is given)")

        self.__src = src_system  # The source anatomical coordinate system
        self.__dst = None  # The desired anatomical coordinate system

        # Mapping from source voxel indices to source coordinate system (either
        # as 4x4 matrix or as quaternion + voxel size + offset information
        self.__vsrc2src = (matrix if matrix is not None else
                           self.__matrix_from_quaternion(*quaternion))

        self.img = img  # The object returned by the file loader
        self.src_volume = voxel_data  # The source voxel data
        self.aligned_spacing = None
        # ^ The voxel sizes / voxel center distances / voxel distances
        # (depending on how one sees a voxel) in the axes and units of the
        # desired destination anatomical coordinate system (as a tuple)
        # [mm/voxel]
        self.aligned_volume = None
        # ^ The transformed voxel data with axes aligned to the desired
        # destination anatomical coordinate system

        # Mapping from the source coordinate system to the destination
        # coordinate system and vice versa (3x3)
        self.__src2dst = None
        self.__dst2src = None

        # Mapping from <src_volume> voxel indices to <aligned_volume> voxel
        # indices and vice versa (4x4)
        self.__vsrc2vdst = None
        self.__vdst2vsrc = None

        # Initialize all the attributes that are currently still None (except
        # for the matrix/quaternion fields that have been left empty)
        self.set_dst_system(dst_system)

    def __init_system_mapping(self):
        """
        Calculate the mapping from the source anatomical coordinate system to
        the currently desired anatomical coordinate system and vice versa.
        """
        self.__src2dst, self.__dst2src = matrix(self.__src, self.__dst)

    def __init_aligned_volume(self):
        """
        Calculate <aligned_volume>: swap the <src_volume> to match the
        currently desired coordinate system.
        """

        ndim = 3
        # Throw away the offset part of the transformation matrix, as it is not
        # relevant here
        mat = self.__vsrc2src[:ndim, :ndim]

        # Find the transformation matrix that maps voxel indices to *original*
        # coordinates without interpolation, i.e. the matrix closest to the
        # given transformation matrix that contains only 0, 1, and -1: For each
        # column, set the element of largest absolute value to +1 or -1
        # according to the element's sign; set all other elements to 0.
        mat_abs = np.abs(mat)
        max_i = np.argmax(mat_abs, axis=0)
        close_mat = np.divide(mat, mat_abs[max_i, range(ndim)])
        close_mat = np.asarray(close_mat, dtype=np.int)

        # Combine: first map the voxels to lie parallel to the original
        # coordinate system's axes, then map to the desired coordinate system.
        # This results in the mapping from the <src_volume> voxel coordinates
        # to <aligned_volume> voxel coordinates (3x3)
        vsrc2vdst3 = np.asarray(np.round(np.dot(self.__src2dst, close_mat)),
                                dtype=int)
        # Make it a 4x4 matrix: Add offset of (dimension size - 1) for the
        # inverted dimensions (in this way account for the inverted and thus
        # negative indices)
        off = np.subtract(self.src_volume.shape, 1).reshape(ndim, 1)
        off = np.dot(vsrc2vdst3, off).clip(max=0)
        vsrc2vdst4 = np.eye(ndim + 1)
        vsrc2vdst4[:ndim, :ndim] = vsrc2vdst3
        vsrc2vdst4[:ndim, ndim] = -off.flatten()
        vdst2vsrc4 = np.asarray(np.round(np.linalg.inv(vsrc2vdst4)), dtype=int)
        validate_swap_matrix(vsrc2vdst4, ndim=ndim)
        validate_swap_matrix(vdst2vsrc4, ndim=ndim)
        # Perform the mapping
        self.aligned_volume = swap(self.src_volume, vsrc2vdst4, ndim)
        self.__vdst2vsrc = vdst2vsrc4
        self.__vsrc2vdst = vsrc2vdst4

    def __init_spacing(self, decimals=3, tol=1e-5):
        """
        Calculate the voxel spacing for the aligned volume.

        If desired (i.e., tol > 0), perform some cleanup for rounding errors:

        If, for a dimension's spacing s,

            abs(round(s, decimals) - s) < tol

        holds, use the rounded value as the spacing for this dimension.
        """
        m = self.get_aligned_matrix()
        # Calculate spacing for all dimensions ...
        d0 = np.linalg.norm(m[0:3, 0])
        d1 = np.linalg.norm(m[0:3, 1])
        d2 = np.linalg.norm(m[0:3, 2])
        # ... then check for round-off errors and select the appropriate values
        unrounded = np.array([d0, d1, d2])
        rounded = np.round(unrounded, decimals)
        use_rounded = abs(rounded - unrounded) < tol
        result = (use_rounded * rounded +
                  np.logical_not(use_rounded) * unrounded)
        self.aligned_spacing = tuple(result)

    def __matrix_from_quaternion(self, q, s, o):
        """
        Calculate the transformation matrix from the <src_volume> voxel
        coordinates to the source coordinate system based on the quaternion <q>
        + voxel size <s> + offset information <0>, return the result as a (4,
        4)-shaped Numpy array.
        """
        # 1. Get matrix from quaternion
        m3 = quaternions.quat2mat(q)
        # 2. Adjust for voxel sizes
        m3 = np.dot(m3, np.diag(s))
        # 3. Combine with offset (3x3 -> 4x4)
        m4 = np.eye(4)
        m4[:-1, :-1] = m3
        m4[:3, 3] = o
        return m4

    def get_src_system(self):
        """
        Get a three-character string that represents the source anatomical
        coordinate system.
        """
        return self.__src

    def get_dst_system(self):
        """
        Get a three-character string that represents the currently desired
        anatomical coordinate system.
        """
        return self.__dst

    def set_dst_system(self, system):
        """
        Set the currently desired destination anatomical coordinate system to
        the given <system> (three-character string expected).
        """
        if system != self.__dst:
            self.__dst = system
            self.__init_system_mapping()
            self.__init_aligned_volume()
            self.__init_spacing()

    def get_src_matrix(self, system=None):
        """
        Get a transformation matrix that maps from <src_volume>'s voxel indices
        to the given coordinate <system> (defaults to the currently desired
        destination coordinate system).

        Return the transformation matrix as a (4, 4)-shaped Numpy array.
        """
        # Get the mapping from the source coordinate system to the desired
        # coordinate system (3x3 -> expand to 4x4)
        src2sys3 = (self.__src2dst if system is None else
                    matrix(self.__src, system)[0])
        src2sys4 = np.eye(4)
        src2sys4[:-1, :-1] = src2sys3
        # Get the mapping from the <src_volume>'s voxel indices to the source
        # coordinate system, combine the two mappings and return
        result = np.dot(src2sys4, self.__vsrc2src)
        return result

    def get_aligned_matrix(self, system=None):
        """
        Get a transformation matrix that maps from <aligned_volume>'s voxel
        indices to the given coordinate <system> (defaults to the currently
        desired destination coordinate system).

        Return the transformation matrix as a (4, 4)-shaped Numpy array.
        """
        # Get the mapping from <aligned_volume> voxel indices to <src_volume> voxel
        # indices (4x4)
        vdst2vsrc = self.__vdst2vsrc
        # Get the mapping from <src_volume> voxel indices to the desired
        # coordinate system (4x4)
        vsrc2sys = self.get_src_matrix(system=system)
        # Combine the two mappings and return
        result = np.dot(vsrc2sys, vdst2vsrc)
        return result

    def equals_content_of(self, other):
        """
        Compare current instance with <other> instance on a voxel level; return
        True if all voxels of the aligned image data arrays are exactly the
        same, return False otherwise.
        """
        # Temporarily align current volume with <other>
        self_system = self.get_dst_system()
        other_system = other.get_dst_system()
        aligned = (self_system == other_system)
        if not aligned:
            self.set_dst_system(other_system)

        # Actually compare the image data
        equal = np.all(self.aligned_volume == other.aligned_volume)

        # Readjust current volume's alignment
        if not aligned:
            self.set_dst_system(self_system)

        return equal


def __repair_dim(data, verbose):
    """
    For 4d <data> arrays with the last two dimensions containing only one
    element, return a new 2d array of the same content.

    For 4d <data> arrays with only the last dimension containing only one
    element, return a new 3d array of the same content.

    For other arrays, simply return them.
    """
    if data.ndim == 4:
        if data.shape[2] == 1 and data.shape[3] == 1:
            # Create and return 2d array
            new_array = np.empty(data.shape[:2], dtype=data.dtype)
            new_array[:] = data[..., 0, 0]
            data = new_array
            if verbose:
                print("Malformed array has been corrected.")
        elif data.shape[3] == 1:
            # Create and return 3d array
            new_array = np.empty(data.shape[:3], dtype=data.dtype)
            new_array[:] = data[..., 0]
            data = new_array
            if verbose:
                print("Malformed array has been corrected.")
    return data


def open_image(path, repair=False, verbose=True):
    """
    Open a 3D NIfTI-1 image at the given <path>.

    Return a <Volume> instance with its <aligned_volume> aligned
    to RAS coordinates.

    Note that the <Volume.src_volume> Numpy array may differ from the array
    that may be obtained via <Volume.img.get_data()>, as an additional step for
    correction of wrong dimensionality may be applied. The correction step is
    triggered by setting <repair> to True (defaults to False). Further note
    that in this case it has not been tested whether the coordinate
    transformations from the NIfTI-1 header still apply.
    """
    # According to the NIfTI-1 specification [2], the world coordinate system
    # of NIfTI-1 files is always RAS.
    src_system = "RAS"

    img = nib.load(path)

    voxel_data = np.array(img.get_data())
    hdr = img.get_header()

    img_dim = hdr["dim"][0]
    if img_dim != 3:
        raise IOError("Currently only 3D images can be handled." +
                      "The given image has %s dimension(s)." % img_dim)

    if verbose:
        print("Image loaded:", path)
        print("Meta data:")
        print(hdr)
        print("Image dimensions:", voxel_data.ndim)

    # Repair corrupted NiBabel import
    if repair:
        voxel_data = __repair_dim(voxel_data, verbose)

    # Create new <Volume> instance

    # Get quaternion + zoom + offset information
    q = hdr.get_qform_quaternion()
    s = hdr.get_zooms()[:3]  # Discard the last value for 4D data
    o = np.array([hdr["qoffset_x"], hdr["qoffset_y"], hdr["qoffset_z"]])
    # Adjust the zoom according to the "qfac" stored in pixdim[0] (cf. [2])
    qfac = hdr["pixdim"][0]
    qfac = qfac if qfac in [-1, 1] else 1
    s = np.multiply(s, (1, 1, qfac))
    # Create and return <Volume>
    return Volume(voxel_data, img, quaternion=(q, s, o), src_system=src_system)


class DataCollection(object):
    '''Abstract class for all data handling classes. 

        Args:
            fetch_fresh_data (bool): Switch to determine if we need to fetch 
                fresh data from the server
    '''

    def __init__(self, **kw):
        self.origargs = copy.deepcopy(kw)
        self.randomstate = np.random.RandomState(argget(kw, "seed", 12345678))
        self.nclasses = argget(kw, 'nclasses', 2)

    def setStates(self, states):
        if states is None:
            logging.getLogger('eval').warning('could not reproduce state, setting unreproducable random seed')
            self.randomstate.set_seed(np.random.randint(0, 1000000))
        self.randomstate.set_state(states)

    def getStates(self):
        return self.randomstate.get_state()

    def reset_seed(self, seed=12345678):
        self.randomstate = np.random.RandomState(seed)

    def random_sample(self, **kw):
        '''Randomly samples from our dataset. If the implementation knows 
        different datasets, the dataset string can be used to choose one, if 
        not, it will be ignored. 
        
        Args:
            **kw: batch_size can be set, but has individual default values, 
                  if not. dataset is also optional, and is ignored where train 
                  and test dataset are not distinguished.
                  
        Returns:
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
        
        Returns:
            A shape array of the dimensionality of the data.
            
        '''
        raise Exception("get_data_dims not implemented in {}"
                        .format(self.__class__))

    def sample_all(self, **kw):
        '''Returns all sequences without creating duplicates. Class depending 
        attributes allow for some intervention.
        
        Args:
            batch_size: Number of samples in  batch.
            **kw: Optional set of arguments, depending on class. If max_size is 
            available, the maximum number of samples that will be returned can 
            be set. normalize (bool), if available, decides if the data should 
            be normalized before it is returned. dataset, if available, lets the
            user choose the dataset that is completely sampled.
        
        Returns:
            A generator that iterates through given collection in batches of 
            batch_size
        
        '''
        raise Exception("sample_all not implemented in {}"
                        .format(self.__class__))

    def _one_hot_labels(self, indexlabels, nclasses=None, softlabels=False, zero_out_label=None):
        if nclasses is None:
            nclasses = self.nclasses
        batch_size = indexlabels.shape[0]
        if indexlabels.shape[1] > 1:
            raise Exception('cant have more than one mask yet')
        else:
            indexlabels = indexlabels.reshape([batch_size]
                                              + list(indexlabels.shape[2:]))
        if softlabels:
            # this cant be right. just as it is, it only works correctly for a 2 classes case.
            l = np.zeros([np.prod(indexlabels.shape), nclasses], dtype=np.float32)
            if nclasses == 2:
                l[np.int32(np.arange(np.prod(indexlabels.shape))), 0] = 1 - indexlabels.flatten()
                l[:, 0] = np.clip(l[:, 0], 0, 1)
                l[:, 1] = 1 - l[:, 0]
            else:
                raise logging.getLogger('data').warning('this part of the code, we strongly discourage')
                indexlabels = np.clip(indexlabels, 0, self.nclasses - 1)
                hi = np.ceil(indexlabels) - indexlabels
                lo = 1 - hi
                l[np.int32(np.arange(np.prod(indexlabels.shape))), np.int32(
                    np.floor(indexlabels)).flatten()] = hi.flatten()
                l[np.int32(np.arange(np.prod(indexlabels.shape))), np.int32(
                    np.ceil(indexlabels)).flatten()] = lo.flatten()
        else:
            indexlabels = np.int32(np.round(indexlabels))
            l = np.zeros([np.prod(indexlabels.shape), nclasses], dtype=np.int32)
            l[np.int32(np.arange(np.prod(indexlabels.shape))), indexlabels.flatten()] = 1
        if zero_out_label is not None:
            l[:, zero_out_label] = 0
        return l.reshape(list(indexlabels.shape) + [nclasses])

    def _one_hot_vectorize(self, indexlabels, nclasses=None, zero_out_label=None):
        '''
        simplified onehotlabels method. we discourage using interpolated labels 
        anyways, hence this only allows integer values in indexlabels
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
        comm = "find '" + os.path.join(folder, '') + "' -type d -exec test -e {}/" + featurefiles[0]
        for i in featurefiles[1:]:
            comm += " -a -e {}/" + i
        for i in maskfiles:
            comm += " -a -e {}/" + i
        comm += " \\; -print\n"
        res, err = subprocess.Popen(comm, stdout=subprocess.PIPE, shell=True).communicate()
        if (sys.version_info > (3, 0)):
            # Python 3 code in this block
            return sorted([str(r, 'utf-8') for r in res.split() if r])
        else:
            # Python 2 code in this block
            return sorted([str(r) for r in res.split() if r])
