#!/usr/bin/python

import nibabel as nib
import nrrd
import os.path
import numpy as np


def create_example_nifti_data():
    shape = (100, 100, 40)
    border_edges = [[30, 70], [30, 70], [10, 30]]
    edge_variation = (15, 15, 8)
    rater_variation = (2, 2, 2)
    pats = ["ruedi", "hans", "eva", "micha", "joerg", "maya", "frieda", "anna", "chelsea", "flynn"]
    belongs_to = ['train', 'train', 'train', 'train', 'train', 'val', 'val', 'test', 'test', 'test']
    affine = np.zeros((4, 4))
    affine[0, 2] = 1
    affine[1, 0] = 1
    affine[2, 1] = 1
    affine[3, 3] = 1
    testdatadir = '.'
    print(testdatadir)
    testdatadirnifti = os.path.join(testdatadir, "nifti")
    if os.path.exists(testdatadirnifti):
        print('Files have already been generated. If something is amiss, delete the nifti folder and start again!')
        return
    for f, pat in zip(belongs_to, pats):
        patdir = os.path.join(os.path.join(testdatadirnifti, f), pat)
        if not os.path.exists(patdir):
            os.makedirs(patdir)
        gt_mask = np.zeros(shape)
        gt_borders = [[x[0] + (np.random.random() * 2 - 1) * e,
                       x[1] + (np.random.random() * 2 - 1) * e] for x, e in zip(border_edges, edge_variation)]
        gt_borders = np.uint32(gt_borders)
        gt_mask[
        gt_borders[0][0]:gt_borders[0][1],
        gt_borders[1][0]:gt_borders[1][1],
        gt_borders[2][0]:gt_borders[2][1]
        ] = 1

        for file in ["flair", "mprage", "t2", "pd"]:
            myfile = os.path.join(patdir, file + ".nii.gz")
            if not os.path.exists(os.path.join(patdir, file + ".nii.gz")):
                dat = np.float32(np.random.random(shape)) * np.random.randint(200, 2400) + np.random.randint(200, 800)
                dat += gt_mask * np.random.random(shape) * np.random.randint(200, 400)
                nib.save(nib.Nifti1Image(dat, affine), myfile)
        for file in ["mask1", "mask2"]:
            myfile = os.path.join(patdir, file + ".nii.gz")
            if not os.path.exists(os.path.join(patdir, file + ".nii.gz")):
                dat = np.zeros(shape, dtype=np.uint8)
                rater_borders = [[x[0] + (np.random.random() * 2 - 1) * e,
                       x[1] + (np.random.random() * 2 - 1) * e] for x, e in zip(gt_borders, rater_variation)]
                rater_borders = np.uint32(rater_borders)
                dat[rater_borders[0][0]: rater_borders[0][1],
                rater_borders[1][0]: rater_borders[1][1],
                rater_borders[2][0]: rater_borders[2][1]] = 1
                nib.save(nib.Nifti1Image(dat, affine), myfile)


if __name__ == "__main__":
    create_example_nifti_data()
