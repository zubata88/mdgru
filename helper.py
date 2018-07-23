__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

TINY = 1e-20
import matplotlib

matplotlib.use('Agg')
import numpy as np
import copy
import os, errno
import scipy.linalg as la
from scipy.stats import special_ortho_group as sog
import logging
import urllib.request
import functools


def notify_user(chat_id, token, message='no message'):
    """ Sends a notification when training is completed or the process was killed.

    Given that a telegram bot has been created, and it's api token is known, a chat_id has been opened, and the
    corresponding chat_id is known, this method can be used to be informed if something happens to our process. More
    on telegram bots at https://telegram.org/blog/bot-revolution.
    :param chat_id: chat_id which is used by telegram to communicate with your bot
    :param token: token generated when creating your bot through the BotFather
    :param message: The message to be sent
    """
    try:
        text = urllib.request.urlopen('https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}'
                                      .format(token, chat_id, message)).read()
        logging.getLogger('helper').info('Return value of bot message: {}'.format(text))
    except Exception as e:
        logging.getLogger('helper').warning('Could not send {} to chat {} of token {}'.format(message, chat_id, token))


def lazy_property(function):
    """This function computes a property or simply returns it if already computed."""
    attribute = "_" + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return wrapper


def deprecated(func):
    """ Decorator function to indicate through our logger that the decorated function should not be used anymore
    :param func: function to decorate
    :return: decorated function
    """

    def print_deprecated(x):
        logging.getLogger('helper').info('The following function has been deprecated and should not be used '
                                         'anymore: {}!'.format(func))
        func(x)

    return print_deprecated


def force_symlink(file1, file2):
    """ Tries to create symlink. If it fails, it tries to remove the folder obstructing its way to create one.
    :param file1: path
    :param file2: symlink name
    """
    try:
        os.symlink(file1, file2)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(file2)
            os.symlink(file1, file2)


def argget(dt, key, default=None, keep=False, ifset=None):
    """ Evaluates function arguments.

    It takes a dictionary dt and a key "key" to evaluate, if this key is contained in the dictionary. If yes, return
    its value. If not, return default. By default, the key and value pair are deleted from the dictionary, except if
    keep is set to True. ifset can be used to override the value, and it is returned instead (except if None).
    :param dt: dictionary to be searched
    :param key: location in dictionary
    :param default: default value if key not found in dictionary
    :param keep: bool, indicating if key shall remain in dictionary
    :param ifset: value override.
    :return: chosen value for key if available. Else default or ifset override.
    """
    if key in dt:
        if keep:
            val = dt[key]
        else:
            val = dt.pop(key)
        if ifset is not None:
            return ifset
        else:
            return val
    else:
        return default


def check_if_kw_empty(class_name, kw, module_name):
    """Prints standardised warning if an unsupported keyword argument is used for a given class"""
    if len(kw):
        logging.getLogger(module_name).warning(
            'There were invalid keywords for {}: {}'.format(class_name,
                                                            ",".join(["{}:{}".format(k, v) for k, v in kw.items()])))


def np_arr_backward(matrix, n, k1, k2):
    """ Transforms from block block circulant matrix to filter representation using indices.
    :param matrix: matrix representation of filter
    :param n: number of channels
    :param k1: filter dim 1
    :param k2: filter dim 2
    :return: filter representation
    """
    return matrix.reshape([n, k1 * k2, n, k1 * k2]).transpose([1, 3, 0, 2])[:, 0, :, :].reshape([k1, k2, n, n])


def np_arr_forward(filt, n, k1, k2):
    """ Transforms from filter to block block circulant matrix representation using indices.
    :param filt: filter variable
    :param n: number of channels
    :param k1: filter dimension 1
    :param k2: filter dimension 2
    :return:  matrix representation of filter
    """
    a, b, c, d, n1, n2 = np.ogrid[0:k1, 0:-k1:-1, 0:k2, 0:-k2:-1, 0:n, 0:n]
    return filt[a + b, c + d, n1, n2].transpose([0, 2, 1, 3, 4, 5]).reshape([k1 * k1, k2 * k2, n, n]).transpose(
        [2, 0, 3, 1]).reshape([k1 * k2 * n, k1 * k2 * n])


def _initializer_Q(k1, k2):
    """ Computes a block circulant k1xk1 matrix, consisting of circulant k2xk2 blocks.

    :param k1: Outer convolution filter size
    :param k2: Inner convolution filter size
    :return: block circulant matrix with circulant blocks
    """
    a = np.arange(k1 * k2).reshape([k1, k2, 1, 1])
    bc = np_arr_forward(a, 1, k1, k2)
    to = bc[0, :]
    arr = np.random.random(k1 * k2) - 0.5
    arr = np.asarray([arr[i] if i < to[i] else -arr[to[i]] for i in range(k1 * k2)])
    arr[0] = 0
    skewsymm = np_arr_forward(arr.reshape(k1, k2, 1, 1), 1, k1, k2)
    I = np.eye(k1 * k2)
    return np.float32(np.matmul(la.inv(I + skewsymm), I - skewsymm))


def initializer_W(n, k1, k2):
    """ Computes kronecker product between an orthogonal matrix T and a circulant orthogonal matrix Q.

    Creates a block circulant matrix using the Kronecker product of a orthogonal square matrix T and a circulant
    orthogonal square matrix Q.

    :param n: Number of channels
    :param k1: Outer convolution filter size
    :param k2: Inner convolution filter size
    :return:
    """
    Q = _initializer_Q(k1, k2)
    if n > 1:
        T = sog.rvs(n)
    else:
        return np.float32(Q)
    return np.float32(np.kron(np.float32(T), Q))


def counter_generator(maxim):
    """ Generates indices over multidimensional ranges.

    :param maxim: Number of iterations per dimension
    :return: Generator yielding next iteration
    """
    maxim = np.asarray(maxim)
    count = np.zeros(maxim.shape)
    yield copy.deepcopy(count)
    try:
        while True:
            arr = (maxim - count) > 1
            lind = len(arr) - 1 - arr.tolist()[::-1].index(True)
            count[lind] += 1
            count[lind + 1:] = 0
            yield copy.deepcopy(count)
    except ValueError:
        pass


def compile_arguments(cls, kw, transitive=False):
    """Extracts valid keywords for cls from given keywords and returns the resulting two dicts.

    :param cls: instance or class having property or attribute "_defaults", which is a dict of default parameters.
    :param transitive: determines if parent classes should also be consulted
    :param kw: the keyword dictionary to separate into valid arguments and rest
    """
    kw = copy.copy(kw)
    new_kw = {}
    if transitive:
        for b in cls.__bases__:
            if hasattr(b, '_defaults'):
                temp_kw, kw = compile_arguments(b, kw, transitive=True)
                new_kw.update(temp_kw)
    # compile defaults array from complex _defaults dict:
    defaults = {k: v['value'] if isinstance(v, dict) else v for k, v in cls._defaults.items()}
    new_kw.update({k: argget(kw, k, v) for k, v in defaults.items()})
    return new_kw, kw


def collect_parameters(cls, kw_args={}):
    args = copy.copy(kw_args)
    for b in cls.__bases__:
        if hasattr(b, '_defaults'):
            args = collect_parameters(b, args)
    params = {k: v for k, v in cls._defaults.items() if isinstance(v, dict) and 'help' in v}
    args.update(params)
    return args


def define_arguments(cls, parser):
    if hasattr(cls, 'collect_parameters'):
        args = cls.collect_parameters()
    else:
        args = collect_parameters(cls)
    for k, v in args.items():
        key = k
        kw = {'help': v['help']}

        if 'type' in v:
            kw['type'] = v['type']
        if 'nargs' in v:
            kw['nargs'] = v['nargs']
        elif 'value' in v and isinstance(v['value'], (list, dict)):
            kw['nargs'] = '+'

        propname = copy.copy(key)
        if 'value' in v:
            if isinstance(v['value'], bool):
                if v['value'] == False:
                    kw['action'] = 'store_true'
                else:
                    kw['dest'] = key
                    kw['action'] = 'store_false'
                    if 'invert_meaning' in v and not 'name' in v:
                        propname = kw['invert_meaning'] + key
                    else:
                        propname = "no_" + key
            else:
                kw['default'] = v['value']

        if 'name' in v: #overrides invert_meaning!
            propname = v['name']
            kw['dest'] = key
        props = ["--" + propname]
        if 'short' in v:
            props = ['-' + v['short']] + props
        if 'alt' in v:
            props += ['--' + x for x in v['alt']]
        parser.add_argument(*props, **kw)
    return parser


def harmonize_filter_size(fs, ndim):
    if fs is None:
        return [7 for _ in range(ndim)]
    if len(fs) != ndim:
        if len(fs) == 1:
            fs = [fs[0] for _ in range(ndim)]
        elif fs is None:
            fs = [7 for _ in range(ndim)]
        else:
            print('Filter size and number of dimensions for subvolume do not match!')
            exit(0)
    return fs