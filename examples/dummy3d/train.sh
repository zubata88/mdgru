#!/bin/bash
if [ ! -d `pwd`/nifti ]; then
    python3 create_dummy_data.py
fi
	
# Automatically train and test on the previously created test data. This does not use any form of data augmentation.

python3 ../../mdgru/run.py --datapath `pwd`/nifti --locationtraining train --locationvalidation val --locationtesting test --optionname discardme --modelname discardme -w 20 50 50 --paddingtesting 0 0 0 --windowsizetesting 40 100 100 -p 10 10 10 -f flair.nii.gz t2.nii.gz pd.nii.gz mprage.nii.gz -m mask1.nii.gz --iterations 100 --validate_each 50 --save_each 50 --nclasses 2 --num_threads 4

