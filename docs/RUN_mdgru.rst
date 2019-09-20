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
    
 ::
    
    --datapath: path to folders with train/val/test data
    --locationxxx: name of folder with data for respective purpose
    --optionname: options
    --modelname: name of model
    -w: subvolume size (for each dimension)
    -p: padding size (for each dimension)
    -f: sequences to include
    -m: masks to include
    --interations: number of iterations
    --nclasses: numbber of classe (min. 2)
    --num_threads: number of threads
    --use_pythorch: use PyTorch version (requires GPU)
