
How to install 
''''''''''''''

The code has been developed in Python==3.5.2. It is best to set up a **virtual environment** (e.g. with `conda <https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/>`_) with the mentioned properties in order to develop the deep learning model. For this purpose, install mdgru (together with mvloader) using pip. In addition, make sure you have `CUDA <https://developer.nvidia.com/cuda-90-download-archive>`_/`cuDNN <https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html>`_ installed.

::

    pip install git+https://github.com/gtancev/mdgru.git
    pip install git+https://github.com/spezold/mvloader.git
