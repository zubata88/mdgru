import sys, os

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
from examples.dummy2d.create_dummy_data_2d import create_example_nifti_data_2d, remove_example_nifti_data_2d
from examples.dummy3d.create_dummy_data import create_example_nifti_data, remove_example_nifti_data
from RUN_mdgru import run_mdgru

testdirectory=os.path.dirname(os.path.realpath(__file__))


def test_basic_2d_run_pytorch():
    create_example_nifti_data_2d(shape=(16, 16), border_edges=[[5, 11], [5, 11]],
                                 edge_variation=(2, 2), rater_variation=(1, 1),
                                 testdatadir=testdirectory, datafolder="nifti2d")
    arg_list = ["--datapath", os.path.join(testdirectory, "nifti2d"),
          "--locationtraining", "train",
          "--locationvalidation", "val",
          "--locationtesting", "test",
          "--optionname", "discardme",
          "--modelname", "discardme",
          "-w", "16", "16",
          "--paddingtesting", "0", "2",
          "--windowsizetesting", "16", "9",
          "-p", "0", "0",
          "-f", "flair.nii.gz", "t2.nii.gz", "pd.nii.gz", "mprage.nii.gz",
          "-m", "mask1.nii.gz",
          "--iterations", "3",
          "--test_each", "2",
          "--nclasses", "2",
          "--num_threads", "2",
          "--use_pytorch"
          ]
    run_mdgru(arg_list)
    remove_example_nifti_data_2d(testdatadir=testdirectory, datafolder="nifti2d")


def test_basic_3d_run_pytorch():
    create_example_nifti_data(shape=(16, 16, 16), border_edges=[[5, 11], [5, 11], [5, 11]],
                              edge_variation=(2, 2, 2), rater_variation=(1, 1, 1),
                              testdatadir=testdirectory, datafolder="nifti3d")
    arg_list = ["--datapath", os.path.join(testdirectory, "nifti3d"),
          "--locationtraining", "train",
          "--locationvalidation", "val",
          "--locationtesting", "test",
          "--optionname", "discardme2",
          "--modelname", "discardme2",
          "-w", "16", "16", "16",
          "--paddingtesting", "0", "4", "0",
          "--windowsizetesting", "16", "12", "16",
          "-p", "0", "0", "0",
          "-f", "flair.nii.gz", "t2.nii.gz", "pd.nii.gz", "mprage.nii.gz",
          "-m", "mask1.nii.gz",
          "--iterations", "3",
          "--test_each", "2",
          "--nclasses", "2",
          "--num_threads", "2",
          "--dont_use_tensorboard",
          "--use_pytorch"
          ]
    try:
        run_mdgru(arg_list)
    finally:
        remove_example_nifti_data(testdatadir=testdirectory, datafolder="nifti3d")


def test_basic_2d_run():
    create_example_nifti_data_2d(shape=(16, 16), border_edges=[[5, 11], [5, 11]],
                                 edge_variation=(2, 2), rater_variation=(1, 1),
                                 testdatadir=testdirectory, datafolder="nifti2d")
    arg_list = ["--datapath", os.path.join(testdirectory, "nifti2d"),
          "--locationtraining", "train",
          "--locationvalidation", "val",
          "--locationtesting", "test",
          "--optionname", "discardme",
          "--modelname", "discardme",
          "-w", "16", "16",
          "--paddingtesting", "0", "2",
          "--windowsizetesting", "16", "9",
          "-p", "0", "0",
          "-f", "flair.nii.gz", "t2.nii.gz", "pd.nii.gz", "mprage.nii.gz",
          "-m", "mask1.nii.gz",
          "--iterations", "3",
          "--test_each", "2",
          "--nclasses", "2",
          "--num_threads", "2",
          ]
    run_mdgru(arg_list)
    remove_example_nifti_data_2d(testdatadir=testdirectory, datafolder="nifti2d")


def test_basic_3d_run():
    create_example_nifti_data(shape=(16, 16, 16), border_edges=[[5, 11], [5, 11], [5, 11]],
                              edge_variation=(2, 2, 2), rater_variation=(1, 1, 1),
                              testdatadir=testdirectory, datafolder="nifti3d")
    arg_list = ["--datapath", os.path.join(testdirectory, "nifti3d"),
          "--locationtraining", "train",
          "--locationvalidation", "val",
          "--locationtesting", "test",
          "--optionname", "discardme2",
          "--modelname", "discardme2",
          "-w", "16", "16", "16",
          "--paddingtesting", "0", "4", "0",
          "--windowsizetesting", "16", "12", "16",
          "-p", "0", "0", "0",
          "-f", "flair.nii.gz", "t2.nii.gz", "pd.nii.gz", "mprage.nii.gz",
          "-m", "mask1.nii.gz",
          "--iterations", "3",
          "--test_each", "2",
          "--nclasses", "2",
          "--num_threads", "2",
          "--dont_use_tensorboard"
          ]
    try:
        run_mdgru(arg_list)
    finally:
        remove_example_nifti_data(testdatadir=testdirectory, datafolder="nifti3d")
