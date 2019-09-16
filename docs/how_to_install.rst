
How to Install 
''''''''''''''

Requirements (on ubuntu) can be installed
using the following lines of code. On other systems, use the
corresponding packages. **Make sure to use tensorflow>=1.8 for the TensorFlow backend.**

It's best to use a virtual environment with Python 3.5.

::

    sudo apt-get install cmake python3-pip curl git python3-dicom

    sudo pip3 install --upgrade pip

    # Either with a GPU, and CUDA/CUDNN installed:
    sudo pip3 install "tensorflow-gpu>=1.8"
    
    # Or:
    sudo pip3 install "tensorflow>=1.8"

    sudo pip3 install torch torchvision visdom

    sudo pip3 install nibabel numpy scipy matplotlib pynrrd

    sudo pip3 install scikit-image scikit-learn simpleitk torch visdom

    sudo pip3 install git+https://github.com/spezold/mvloader.git

Or simply install mdgru from github using pip:

::

    pip3 install git+https://github.com/gtancev/mdgru.git
