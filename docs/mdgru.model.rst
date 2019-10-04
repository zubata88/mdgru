Model (Tensorflow backend)
==========================

These are optional model inputs.

**stride (dtype=int)**
::

    --strides None

**use dropconnect on input x (dtype=bool)**
::

    --use_dropconnect_x True
    
**use dropconnect on input h (dtype=bool)**
::

    --use_dropconnect_h True
    
**don't use average pooling (dtype=bool)**
::
    
    --no_avg_pooling True
    
**filter size for input x (dtype=int)**
::

    --filter_size_x 7
    
**filter size for input h (dtype=int)**
::

    --filter_size_h 7
    
**use static RNN (dtype=bool)**
::

    --use_static_rnn False

Subpackages
-----------

.. toctree::

    mdgru.model.crnn
    mdgru.model.mdrnn


MDGRUClassification
------------------------------------------

.. automodule:: mdgru.model.mdgru_classification
    :members:
    :undoc-members:
    :show-inheritance:


Module Contents
---------------

.. automodule:: mdgru.model
    :members:
    :undoc-members:
    :show-inheritance:
