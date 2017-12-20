__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

from tensorflow_extra_ops import caffebicgru
import tensorflow as tf
from helper import get_modified_xavier_method


def CaffeBiCGRU3D(inp,dimension,outC,form,dropconnectx=None,dropconnecth=None,filterx=None,filterh=None,bias=None,fsy=3,fsz=3,favorspeedovermemory=True,uniform_init=False,use_bernoulli_dropconnect=False):
    if form == "NDHWC":
        needs_transpose = True
        inp = tf.transpose(inp,[0,4,1,2,3])
    elif form == "NCDHW":
        needs_transpose = False
    else:
        raise Exception("format {} not known".format(form))
    N,inC,X,Y,Z = inp.get_shape().as_list()
    if N is not None and N > 1:
        raise Exception("cant do that yet...")
    else:
        N = 1
    if dimension < 0 or dimension > 2:
        raise Exception("dimension takes values in {0,1,2}")

    filtersizey = fsy*2+1
    filtersizez = fsz*2+1

    if filterx is None:
        filterx = tf.get_variable("filterx",shape=[2,3,outC,inC,filtersizey,filtersizez],initializer=get_modified_xavier_method(inC*filtersizey*filtersizez,uniform_init))
    if filterh is None:
        filterh = tf.get_variable("filterh",shape=[2,3,outC,outC,filtersizey,filtersizez],initializer=get_modified_xavier_method(outC*filtersizey*filtersizez,uniform_init))
       
    if dropconnectx is not None:
        if use_bernoulli_dropconnect:
            mykeepsx = tf.cast(tf.random_uniform([2,3,outC,inC,filtersizey,filtersizez], 0, 1, tf.float32, None, "mydropconnectx")<dropconnectx,tf.float32)/dropconnectx
        else:
            mykeepsx = tf.random_normal([2,3,outC,inC,filtersizey,filtersizez],1,tf.sqrt((1-dropconnectx)/dropconnectx))
        filterx *= mykeepsx
    if dropconnecth is not None:
        if use_bernoulli_dropconnect:
            mykeepsh = tf.cast(tf.random_uniform([2,3,outC,outC,filtersizey,filtersizez], 0, 1, tf.float32, None, "mydropconnecth")<dropconnecth,tf.float32)/dropconnecth
        else:
            mykeepsh = tf.random_normal([2,3,outC,outC,filtersizey,filtersizez],1,tf.sqrt((1-dropconnecth)/dropconnecth))
        filterh *= mykeepsh
    if bias is None:
        bias = tf.get_variable("bias",shape=[2,3,outC],initializer=tf.constant_initializer())

    inners = [X,Y,Z]
    outer=inners.pop(dimension)
    if favorspeedovermemory:   
        tempshape = [2,N,outC,outer]+inners
    else:
        tempshape = [2,N,outC]+inners
    

    z = tf.get_variable("z",shape=tempshape,initializer=tf.constant_initializer(),trainable=False)
    r = tf.get_variable("r",shape=tempshape,initializer=tf.constant_initializer(),trainable=False)
    deltard = tf.get_variable("deltard",shape=tempshape,initializer=tf.constant_initializer(),trainable=False)
    ht = tf.get_variable("ht",shape=tempshape,initializer=tf.constant_initializer(),trainable=False)
    hd = tf.get_variable("hd",shape=[2,N,outC,outer]+inners,initializer=tf.constant_initializer(),trainable=False)

    with tf.control_dependencies([z,r,deltard,ht,hd,inp,filterx,filterh,bias]):
        res = caffebicgru.caffe_c_g_r_u.caffe_cgru_step_op(inp,z,r,deltard,ht,filterx,filterh,bias,hd,dimension=dimension,outC=outC,inC=inC,X=X,Y=Y,Z=Z,fsy=filtersizey,fsz=filtersizez,favorspeedovermemory=favorspeedovermemory,N=N)
    if needs_transpose:
        res = tf.transpose(res,[0,2,3,4,1])
    return res
