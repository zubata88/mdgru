import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from model.mdgru_classification import MDGRUClassification
import pytest
import tensorflow as tf

def create_dummy_model_tensors_3d(w=[16, 16, 16], nclasses=2, nfeatures=3):
    target = tf.placeholder(dtype=tf.bool, shape=[None] + list(w) + [nclasses])
    dropout = tf.placeholder(dtype=tf.float32)
    data = tf.placeholder(dtype=tf.float32, shape=[None] + list(w) + [nfeatures])
    return data, target, dropout

def test_mdgruclassification_instantiation():
    a = MDGRUClassification(*create_dummy_model_tensors_3d(), kw={})
