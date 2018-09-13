#!/usr/bin/env python

import io
import os
import sys
from shutil import rmtree

from setuptools import setup, find_packages, Command

# Meta Data
NAME = 'dtw'
DESCRIPTION = 'Dynamic Time Warping (DTW) in Python'
URL = 'https://github.com/pierre-rouanet/dtw'
EMAIL = 'pierre.rouanet@gmail.com'
AUTHOR = 'Pierre Rouanet'
# REQUIRES_PYTHON = '>=3.6.0'
VERSION = '1.4'

REQUIRED = [
        'numpy',
        'scipy',
    ]

here = os.path.abspath(os.path.dirname(__file__))

long_description = None
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

setup(name=NAME,
      version=about['__version__'],
      description=DESCRIPTION,
      author=AUTHOR,
      author_email=EMAIL,
      url=URL,
      license='GNU GENERAL PUBLIC LICENSE Version 3',

      install_requires=REQUIRED,
      setup_requires=['setuptools_git >= 0.3', ],
      classifiers=[
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
        ],
      packages=find_packages(exclude=('tests',)),
      #py_modules=['dtw'],
      test_suite='tests'
      )
