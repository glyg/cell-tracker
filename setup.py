# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import warnings

from setuptools import find_packages
from setuptools import setup
from setuptools import Extension

# Get version number
import sys
sys.path.append('.')
import cell_tracker

# Fill project desciption fields
DISTNAME = 'cell-tracker'
DESCRIPTION = 'Application of `scikit-tracker` to track cell clusters'
LONG_DESCRIPTION = open('README.md').read()
MAINTAINER = 'Guillaume Gay'
MAINTAINER_EMAIL = 'gllm.gay@gmail.com'
URL = 'http://bnoi.github.io/cell-tracker'
LICENSE = 'LGPL'
DOWNLOAD_URL = 'https://github.com/bnoi/cell-tracker'
VERSION = cell_tracker.__version__
PYTHON_VERSION = (3, 3)
DEPENDENCIES = ["numpy >= 1.8",
                "scipy >= 0.12",
                "pandas >= 0.13",
                "scikit-image >= 0.9",
                "scikit-learn >= 0.13",
                "sktracker >= 0.1"
                ]

if VERSION.endswith('dev'):
    DEPENDENCIES += ["nose >= 1.3",
                     "sphinx >= 1.2",
                     "coverage >= 3.7"
                     ]

if __name__ == "__main__":

    setup(
        name=DISTNAME,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        url=URL,
        license=LICENSE,
        download_url=DOWNLOAD_URL,
        version=VERSION,

        classifiers=[
            'Development Status :: 2 - Pre-Alpha',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: LGPL License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
        ],

        packages=find_packages(),
        package_data={
            '': ['data/*.h5', 'data/*.xlsx', 'data/*.tif'],
        },

        tests_require='nose',
        test_suite='nose.collector',

        # Should DEPENDENCIES need to be included or let the user install them himself ?
        install_requires=[],
        # install_requires=DEPENDENCIES,
        setup_requires=['numpy'],
    )
