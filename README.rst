Multi-dimensional Gated Recurrent Units
=======================================

This repository contains the code used to generate the result in the
paper *Automated Segmentation of Multiple Sclerosis Lesions using
Multi-Dimensional Gated Recurrent Units*. It is implemented in **Python** using the deep learning libraries **PyTorch** and **TensorFlow** each, modified versions were also
used to reach *1st place* in the ISBI 2015 longitudinal lesion
segmentation challenge, *2nd place* in the white matter hyperintensities
challenge of MICCAI 2017 (and its previous implementation using Caffe made
*3rd place* in the MrBrainS13 Segmentation challenge). 
It was also applied in the BraTS 2017 competition, where the information on the exact rank are still
unknown.

Since being published the first time using a Caffe implementation, the code has been improved on quite a
bit, especially to facilitate handling training and testing runs. The
reported results should still be reproducible though using this new
implementation with TensorFlow and PyTorch. (The former Caffe code is not maintained anymore (there are probably breaking
changes in CuDNN, not tested), but a snapshot of it is included in this
release in the folder tensorflow\_extra\_ops as additional operation for
TensorFlow.)

The code has been developed in Python==3.5.2. It is best to set up a **virtual environment** (e.g. with `conda <https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/>`_) with the mentioned properties in order to develop the deep learning model. For this purpose, follow the instructions in the `docs <https://mdgru.readthedocs.io/en/latest/index.html>`_, and install mdgru (together with mvloader) using pip.

::

    pip3 install git+https://github.com/gtancev/mdgru.git
    pip3 install git+https://github.com/spezold/mvloader.git

Usage on a high performance computing (HPC) cluster
'''''''''''''''''''''''''''''''''''''''''''''''''''
The slurm submission file should look like this:

::

    #!/bin/bash

    #SBATCH --job-name=mdgru
    #SBATCH --cpus-per-task=1
    #SBATCH --mem-per-cpu=8G
    #Total memory reserved: 8GB
    #SBATCH --partition=k80     
    #SBATCH --gres=gpu:2 

    #SBATCH --time=00:30:00
    #SBATCH --qos=30min

    # Paths to STDOUT or STDERR files should be absolute or relative to current working directory
    #SBATCH --output=stdout
    #SBATCH --mail-type=END,FAIL,TIME_LIMIT
    #SBATCH --mail-user=your.email@adress.com

    #This job runs from the current working directory

    #Remember:
    #The variable $TMPDIR points to the local hard disks in the computing nodes.
    #The variable $HOME points to your home directory.
    #The variable $JOB_ID stores the ID number of your task.

    #load your required modules below
    #################################
    ml Python/3.5.2-goolf-1.7.20
    ml CUDA/9.0.176
    ml cuDNN/7.3.1.20-CUDA-9.0.176

    #export your required environment variables below
    #################################################
    source "/pathtoyourfolderbeforeanaconda3/anaconda3/bin/activate" nameofvirtualenvironment

    #add your command lines below
    #############################
    python3 RUN_mdgru.py --datapath files --locationtraining train \
    --locationvalidation val --locationtesting test \
    --optionname defaultsettings --modelname mdgrudef48 -w 64 64 64 -p 5 5 5 \
    -f pd_pp.nii t2_pp.nii flair_pp.nii mprage_pp.nii -m mask.nii --iterations 10 \
    --nclasses 2 --num_threads 4

Papers
''''''

Reference implementation (and based on former Caffe version):

::

    @inproceedings{andermatt2016multi,
      title={Multi-dimensional gated recurrent units for the segmentation of biomedical 3D-data},
      author={Andermatt, Simon and Pezold, Simon and Cattin, Philippe},
      booktitle={International Workshop on Large-Scale Annotation of Biomedical Data and Expert Label Synthesis},
      pages={142--151},
      year={2016},
      organization={Springer}
    }

Code also used for (with modifications):

::

    @inproceedings{andermatt2017a,
      title = {{{Automated Segmentation of Multiple Sclerosis Lesions}} using {{Multi-Dimensional Gated Recurrent Units}}},
      timestamp = {2017-08-09T07:27:10Z},
      journal = {Lecture Notes in Computer Science},
      author = {Andermatt, Simon and Pezold, Simon and Cattin, Philippe},
      year = {2017},
      booktitle={International Workshop on Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries},
      note = {{{[accepted]}}},
      organization={Springer}
    }
    
    @article{andermatt2017b,
      title={Multi-dimensional Gated Recurrent Units for Automated Anatomical Landmark Localization},
      author={Andermatt, Simon and Pezold, Simon and Amann, Michael and Cattin, Philippe C},
      journal={arXiv preprint arXiv:1708.02766},
      year={2017}
    }
    
    @article{andermatt2017wmh,
      title={Multi-dimensional Gated Recurrent Units for the Segmentation of White Matter Hyperintensites},
      author={Andermatt, Simon and Pezold, Simon and Cattin, Philippe}
    }
    
    @inproceedings{andermatt2017brats,
    title = {Multi-dimensional Gated Recurrent Units for
    Brain Tumor Segmentation},
    author = {Simon Andermatt and Simon Pezold and Philippe C. Cattin},
    year = 2017,
    booktitle = {2017 International {{MICCAI}} BraTS Challenge}
    }

When using this code, please cite at least *andermatt2016multi*, since
it is the foundation of this work. Furthermore, feel free to cite the
publication matching your use-case from above. E.g. if you're using the
code for pathology segmentation, it would be adequate to cite
*andermatt2017a* as well.

Acknowledgements
''''''''''''''''

We thank the Medical Image Analysis Center for funding this work. |MIAC
Logo|

.. |MIAC Logo| image:: http://miac.swiss/gallery/normal/116/miaclogo@2x.png

