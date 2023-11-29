#!/usr/bin/env python

"""
Creates python package for w4c23
"""

from setuptools import find_packages, setup

setup(
    name="w4c23",
    version="1.0",
    description="Weather4Cast 2023",
    author="Rafael Pablos Sarabia",
    author_email="",
    url="",
    install_requires=["pytorch-lightning"],
    packages=find_packages(),
)
