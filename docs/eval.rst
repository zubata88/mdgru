Evaluation module
=================

Supervised Evaluation
---------------

.. automodule:: mdgru.eval 
    :members:
    :undoc-members:

Tensorflow Backend
------------------

.. automodule:: mdgru.eval.tf
    :members:
    :undoc-members:
    
**save only labels and no probability distributions (dtype=bool)**
::
    
    --only_save_label False
    
**always pick other random samples for validation (dtype=bool)**
::
    
    --validate_same True
    
**number times we want to evaluate one volume; this only makes sense using a keep rate of less than 1 during evaluation (dropout_during_evaluation less than 1) (dtype=int)**

::

    --evaluate_uncertainty_times 1
    
**keep rate of weights during evaluation; useful to visualize uncertainty in conjunction with a number of samples per volume (dtype=float)**

::

    --evaluate_uncertainty_dropout 1.0
    
**Save each evaluation sample per volume. Without this flag, only the standard deviation and mean over all samples is kept. (dtype=bool)**

::

    --evaluate_uncertainty_saveall False

PyTorch Backend
---------------

.. automodule:: mdgru.eval.torch
    :members:
    :undoc-members:
