Data loader module
==================


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
    
**correct nifti orientation (dtype=bool)**
::
    
    --correct_orientation True

DataCollection
---------------

.. automodule:: mdgru.data
    :members:
    :undoc-members:

GridDataCollection
------------------

.. automodule:: mdgru.data.grid_collection
    :members:
    :undoc-members:
