# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CaffeMDGRU op Python library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import ops

import sys
import os.path
import subprocess
import tensorflow as tf

currdir = os.path.dirname(os.path.abspath(__file__))

if (sys.version_info > (3, 0)):
    # Python 3 code in this block
    
    if not os.path.exists(os.path.join(currdir,'caffe_c_g_r_u3.so')):
        #compile:
        subprocess.call(['make','-C',currdir,'caffecgru3'])
    
    caffe_c_g_r_u = tf.load_op_library(
         os.path.join(tf.resource_loader.get_data_files_path(),
                'caffe_c_g_r_u3.so'))
    caffe_c_g_r_u_grad = tf.load_op_library(
         os.path.join(tf.resource_loader.get_data_files_path(),
                'caffe_c_g_r_u_gradient3.so'))
else:
    # Python 2 code in this block
    if not os.path.exists(os.path.join(currdir,'caffe_c_g_r_u2.so')):
        #compile:
        subprocess.call(['make','-C',currdir,'caffecgru2'])
    caffe_c_g_r_u = tf.load_op_library(
         os.path.join(tf.resource_loader.get_data_files_path(),
                'caffe_c_g_r_u2.so'))
    caffe_c_g_r_u_grad = tf.load_op_library(
         os.path.join(tf.resource_loader.get_data_files_path(),
                'caffe_c_g_r_u_gradient2.so'))



@ops.RegisterGradient("CaffeCGRUStepOp")
def caffe_cgru_step_grad(op,grad):
    x = op.inputs[0]
    z = op.inputs[1]
    r = op.inputs[2]
    deltar = op.inputs[3] 
    ht = op.inputs[4]
    filterx =op.inputs[5]
    filterh =op.inputs[6]
    bias = op.inputs[7]
    hd =op.inputs[8]
    deltah = grad

    #FIXME: IMPLEMENT THIS TOMORROW MORNING!!
    with tf.control_dependencies([x,z,r,deltar,ht,filterx,filterh,bias,hd,deltah]):
        dx,dfx,dfh,db = caffe_c_g_r_u_grad.caffe_cgru_gradient_step_op(x,z,r,deltar,ht,filterx,filterh,bias,hd,deltah,
                    dimension=op.get_attr('dimension'),
                    outC=op.get_attr('outC'),inC=op.get_attr('inC'),X=op.get_attr('X'),
                    Y=op.get_attr('Y'),Z=op.get_attr('Z'),
                    fsy=op.get_attr('fsy'),fsz=op.get_attr('fsz'),
                    favorspeedovermemory=op.get_attr('favorspeedovermemory'),
                    N=op.get_attr('N'))
#     dfx = tf.Print(dfx,[dfx],'dfx')
#     dfh = tf.Print(dfh,[dfh],'dfh')
#     db = tf.Print(db,[db],'db')
#     dx = tf.Print(dx,[dx],'dx')
    #if not op.get_attr('favorspeedovermemory'):
    #    deltahd = None
    return [dx,None,None,None,None,dfx,dfh,db,None]



