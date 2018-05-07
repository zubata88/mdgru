#/bin/bash
# Automatically find last checkpoint file (first line), and evaluate that model on the test data (second line)

ckpt=$(ls `pwd`/nifti/experiments/mdgru_discardme/*/cache/*.ckpt-*.index -dArt | grep -v -e "latest" | tail -n 1 | sed 's/.index//g')

python3 ../../RUN_mdgru.py --datapath `pwd`/nifti --onlytest --locationtesting test --optionname discardme -p 0 0 0 -w 40 100 100 -f flair.nii.gz t2.nii.gz pd.nii.gz mprage.nii.gz -m mask1.nii.gz --nclasses 2 --num_threads 4 --ckpt $ckpt 

