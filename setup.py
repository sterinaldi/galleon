import numpy
from setuptools import setup, find_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
from codecs import open
from os import path
from distutils.extension import Extension
import os
import warnings

try:
    from Cython.Build import cythonize
except ImportError:
    raise ImportError("Cython not found. Please install it via\n\tpip install Cython")

with open("requirements.txt") as requires_file:
    requirements = requires_file.read().split("\n")
with open("README.md") as readme_file:
    long_description = readme_file.read()

# see https://stackoverflow.com/a/21621689/1862861 for why this is here
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        self.include_dirs.append(numpy.get_include())

setup(
    name = 'galleon',
    description = 'Synthetic GW posterior catalog generator',
    author = 'Stefano Rinaldi',
    author_email = 'stefano.rinaldi@phd.unipi.it',
    url = 'https://github.com/sterinaldi/galleon',
    python_requires = '>=3.7',
    packages = ['galleon'],
    install_requires=requirements,
    include_dirs = ['galleon', numpy.get_include()],
    setup_requires=['numpy', 'cython'],
    package_data={"": ['*.c', '*.pyx', '*.pxd']},
    version='1.0.0',
    long_description=long_description,
    long_description_content_type='text/markdown',
    )
