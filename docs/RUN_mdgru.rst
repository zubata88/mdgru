Start script
============

.. automodule:: RUN_mdgru
    :members:
    :undoc-members:

Below is a sample start script with minimum inputs.

::

    python3 RUN_mdgru.py --datapath path/to/samplestructure --locationtraining train_data \
    --locationvalidation val_data --locationtesting test_data \
    --optionname defaultsettings --modelname mdgrudef48 -w 64 64 64 -p 5 5 5 \
    -f seq1.nii.gz seq2.nii.gz -m lab.nii.gz --iterations 100000 \
    --nclasses 4
    
Mandatory inputs
----------------

These are mandatory inputs for which no default values are set.

**path to folders with train/val/test data (dtype=str)**
::

    --datapath path

**name of folder with data for respective purpose (dtype=str)**
::
    
    --locationtraining foldernametrain --locationvalidation foldernameval --locationtesting foldernametest

**name of options/settings (dtype=str)**
::

    --optionname bestoptionsever

**name of model (dtype=str)**
::
    
    --modelname bestmodelever

**subvolume size (dtype=int) of shape (1, n_dims)**
::
    
    -w 128 128 128

**padding size (dtype=int) of shape (1, n_dims)**
::
    
    -p 5 5 5

**sequences to include (dtype=str)**
::
    
    -f t2.nii flair.nii
    
**masks to include (dtype=str)**
::
    
    -m mask1.nii

**iterations of training to perform (dtype=int); only possible if --epochs 1**
::
    
    --interations 100000

**number of classes (dtype=int)**
::

    --nclasses 2

Optional inputs
---------------

These are optional inputs for which the default values (listed in the commands below) can be changed manually.


**number of threads in data collection for data prefetching (dtype=int)**
::
    
    --num_threads 3
    
**non-threaded data collection (dtype=int)**
::
    
    --nonthreaded

**epochs to perform (dtype=int); does not work together with iterations**
::
    
    --epochs 1

**use PyTorch version (dtype=bool)**
::
    
    --use_pytorch False
    
**use only CPU (dtype=bool)**

::
    
    --only_cpu False

**fraction of GPU memory to use (dtype=float) in [0,1]**
::    
    
    --gpubound 1.0
    

**probability for dropout (dtype=float) in [0,1]**
::  

    --dropout_rate 0.5

**use batch normalization (dtype=bool)**
::  

    --add_e_bn False
    
**use skip connections/residual learning; add a residual connection around a MDGRU block (dtype=bool)**
::  

    --resmdgru False

**data augmentation: standard deviation to use for Gaussian filtered images during high pass filtering (dtype=int)**
::

    --SubtractGaussSigma 5
    
**data augmentation: use only Gauss-Sigma filtered images (dtype=bool)**
::

    --nooriginal False

**data augmentation: deformation grid spacing in pixels (dtype=int); if zero: no deformation will be applied**
::
    
    --deform 0

**data augmentation: given a deformation grid spacing, this determines the standard deviations for each dimension of the random deformation vectors (dtype=float)**
::

    --deformSigma 0.0

**data augmentation: activate random mirroring along the specified axes during training (dtype=bool)**
::
    
    --mirror False

**data augmentation: random multiplicative Gaussian noise with unit mean, unit variance (dtype=bool)**
::
    
    --gaussiannoise False

**data augmentation: amount of randomly scaling images per dimension as a factor (dtype=float)**
::
    
    --scaling 0.0

**data augmentation: amount in radians to randomly rotate the input around a randomly drawn vector (dtype=float)**
::

    --rotation 0.0

**sampling outside of discrete coordinates (dtype=float)**
::

    --shift 0.0

**interpolation when using no deformation grids (dtype=bool)**
::

    --interpolate_always False

**define random seed for deformation variables (dtype=int)**
::

    --deformseed 1234

**spline order interpolation_order in 0 (constant), 1 (linear), 2 (cubic) (dtype=int)**
::

    --interpolation_order 3

**rule on how to add values outside image boundaries ("constant", "nearest", "reflect", "wrap") (dtype=str)**
::

    --padding_rule constant

**whiten image data to mean 0 and unit variance (dtype=bool)**
::

    --whiten True

**force each n-th sample to contain labelled data (dtype=int)**
::

    --each_with_labels 0
    
**whether channels appear first (PyTorch) or last (TensorFlow) (dtype=bool)**
::

    --channels_first False

**if multiple masks are provided, we select one at random for each sample (dtype=bool)**
::

    --choose_mask_at_random False

**perform one-hot-encoding from probability distributions (dtype=bool)**
::

    --perform_one_hot_encoding True

**ignore missing masks (dtype=bool)**
::
    
    --ignore_missing_mask False

