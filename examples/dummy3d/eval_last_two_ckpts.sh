#!/bin/bash
# Automatically select last two checkpoints (first line) and evaluate them using automatic naming using one optionname (second line)

ckpts=$(ls `pwd`/nifti/experiments/mdgru_discardme/*/cache/*.ckpt-*.index -dArt | grep -v -e "latest" | tail -n 2 | sed 's/.index//g')

python3 ../../RUN_mdgru.py --datapath `pwd`/nifti --onlytest --locationtesting test --optionname discardme -p 0 0 0 -w 40 100 100 -f flair.nii.gz t2.nii.gz pd.nii.gz mprage.nii.gz -m mask1.nii.gz --nclasses 2 --num_threads 4 --ckpt $ckpts 

