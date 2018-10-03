Multi-dimensional Gated Recurrent Units
=======================================

This repository contains the code used to produce the results in the
papers **Automated Segmentation of Multiple Sclerosis Lesions using
Multi-Dimensional Gated Recurrent Units** and **Multi-dimensional Gated
Recurrent Units for Automated Anatomical Landmark Localization**. It was
used to reach *1st place* in the ISBI 2015 longitudinal lesion
segmentation challenge, *2nd place* in the white matter hyperintensities
challenge of MICCAI 2017 and its previous implementation in CAFFE made
*3rd place* in the MrBrainS13 Segmentation challenge. We also competed
in BraTS 2017, where the information on the exact rank are still
unknown. This release is implemented using the tensorflow framework. The
CAFFE code is not maintained anymore (there are probably breaking
changes in CuDNN, not tested), but a snapshot of it is included in this
release in the folder tensorflow\_extra\_ops as additional operation for
tensorflow. Since being published, the code has been improved on quite a
bit, especially to facilitate handling training and testing runs. The
reported results should still be reproducible though using this
implementation. 

The latest full documentation is avaliable `here <https://zubata88.github.io/mdgru>`_.

Papers
''''''

Reference implementation for - and based on - Caffe version:

::

    @inproceedings{andermatt2016multi,
      title={Multi-dimensional gated recurrent units for the segmentation of biomedical 3D-data},
      author={Andermatt, Simon and Pezold, Simon and Cattin, Philippe},
      booktitle={International Workshop on Large-Scale Annotation of Biomedical Data and Expert Label Synthesis},
      pages={142--151},
      year={2016},
      organization={Springer}
    }

Code used for:

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

