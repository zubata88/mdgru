
How to Install 
''''''''''''''

Requirements (on ubuntu) can be installed
using the following lines of code. On other systems, use the
corresponding packages. Make sure to use tensorflow v 1.8 for the tensorflow backend.

::

    sudo apt-get install cmake python3-pip curl git python3-dicom

    sudo pip3 install --upgrade pip

    # either with a gpu, and cuda + cudnn installed:
    sudo pip3 install "tensorflow-gpu>=1.8"
    # or
    sudo pip3 install "tensorflow>=1.8"

    sudo pip3 install torch torchvision visdom

    sudo pip3 install nibabel numpy scipy matplotlib pynrrd scikit-image scikit-learn

    sudo pip3 install git+https://github.com/spezold/mvloader.git

Or simply install mdgru from github using pip:

::

    pip3 install git+https://github.com/zubata88/mdgru.git
