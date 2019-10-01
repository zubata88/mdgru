Start Script
============

.. automodule:: RUN_mdgru
    :members:
    :undoc-members:

::

    python3 RUN_mdgru.py --datapath path/to/samplestructure --locationtraining train_data \
    --locationvalidation val_data --locationtesting test_data \
    --optionname defaultsettings --modelname mdgrudef48 -w 64 64 64 -p 5 5 5 \
    -f seq1.nii.gz seq2.nii.gz -m lab.nii.gz --iterations 100000 \
    --nclasses 4 --num_threads 4 (--use_pytorch)
    
Mandatory Inputs
----------------

--datapath: str, path to folders with train/val/test data
--locationxxx: str, name of folder with data for respective purpose
--optionname: str, options/settings
--modelname: name of model
-w: int of shape n_dims, subvolume size (for each dimension)
-p: int of shape n_dims, padding size (for each dimension)
-f: str, sequences to include
-m: str, masks to include
--interations: int, number of iterations
--nclasses: int, number of classes (min. 2)
--num_threads: int, number of threads

Optional Inputs
---------------

--use_pythorch: bool (default: False), use PyTorch version (requires GPUs)
--dropbout_rate: float [0,1] (default: 0.5), probability for dropout_rate
--only_cpu: bool (default: False) use only only_cpu
--gpubound: float [0,1] (default: 1.0), fraction of GPU memory to use
--SubtractGaussSigma: int (default: 5), data augmentation, standard deviation to use for Gaussian filtered images during high pass filtering
--nooriginal: bool (default: False), use only Gauss-Sigma filtered images
--deform: int (default: 0), deformation grid spacing in pixels, if zero, no deformation will be applied
--deformSigma: float (default: 0.0), given a deformation grid spacing, this determines the standard deviations for each dimension of the random deformation vectors
--mirror: bool (default: False), activate random mirroring along the specified axes during training
--gaussiannoise: bool (default: False), random mult. Gaussian noise (unit mean, unit variance)
--scaling: float (default: 0.0), amount of randomly scaling images (per dimension) as a factor (e.g. 1.5)
--rotation: float (default: 0.0), amount in radians to randomly rotate the input around a randomly drawn vector
--shift:  float (default: 0.0), sampling outside of discrete coordinates
--interpolate_always: bool (default: False), interpolate when using no deformation grids
--deformseed: int (default: 1234), define random seed for deformation variables
--interpolation_order: int (default: 3), spline order interpolation_order (0 (constant), 1 (linear), 2 (cubic))
--padding_rule: str (default: "constant"), rule on how to add values outside image boundaries ("constant", "nearest", "reflect", "wrap")
--whiten: bool (default: True), whiten image data to mean 0 and unit variance
--each_with_labels: int (default: 0), force each n-th sample to contain labelled data
--channels_first: bool (default: False), whether channels appear first (PyTorch) or last (TensorFlow)
--choose_mask_at_random: bool (default: False),if multiple masks are provided, we select one at random for each sample
--perform_one_hot_encoding: bool (default: True)
--ignore_missing_mask: bool (defaul: False)
