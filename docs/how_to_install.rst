
How to install 
''''''''''''''

Requirements (on ubuntu) can be installed
using the following lines of code. Dependencies will be handled by the setup.py file (installation via GitHub using pip). In addition, **mvloader** has to be installed.


The code has been developed in Python==3.5.2. It is best to set up a **virtual environment** (e.g. with `conda <https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/>`_) with the mentioned properties in order to develop the deep learning model. For this purpose, follow the instructions in the `docs <https://mdgru.readthedocs.io/en/latest/index.html>`_, and install mdgru (together with mvloader) using pip. In additon, make sure you have CUDA/cuDNN installed.

::

    pip3 install git+https://github.com/gtancev/mdgru.git

    pip3 install git+https://github.com/spezold/mvloader.git
