python3 ../../RUN_mdgru.py --datapath `pwd`/nifti --locationtraining train --locationvalidation val --locationtesting test --optionname discardme --modelname discardme -w 256 256 --paddingtesting 10 10 --windowsizetesting 256 256 -p 0 0 -f flair.nii.gz t2.nii.gz pd.nii.gz mprage.nii.gz -m mask1.nii.gz --iterations 100 --validate_each 50 --nclasses 2 --num_threads 4

