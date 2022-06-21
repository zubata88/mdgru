#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

print(find_packages('.'))
setup(
    name="mdgru",
    version="0.2.20181002",
    description="Segmentation Suite for multi-dimensional gated recurrent units (MDGRU)",
    long_description="Details can be found in the Readme file",
    author="Simon Andermatt",
    author_email="simon.andermatt@unibas.ch",
    url="https://github.com/zubata88/mdgru",
    packages=find_packages('.') + ['.'],
    license="LGPL",
    python_requires='>=3.5',
    install_requires=["nibabel==2.5.0", "numpy==1.22.0", "scipy==1.0.0", "pydicom==1.3.0", "matplotlib==3.0.3", "scikit-image==0.15.0", "simpleitk==1.2.2", "tensorflow-gpu==1.8", "torch==1.2.0", "torchvision==0.4.0", "visdom==0.1.8.9", "imageio==2.13.5"],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)',
        'Programming Language :: Python :: 3',
    ],
)
