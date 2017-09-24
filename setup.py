#!/usr/bin/env python3

from setuptools import setup
from glob import glob

setup(
    name="saddle",
    version="0.1",
    description="Geometry optimization program for chemical reaction",
    license='GPLv3',
    author='Derrick Yang',
    author_email='yxt1991@gmail.com',
    package_dir={'saddle': 'saddle'},
    packages=[
        'saddle', 'saddle.test', 'saddle.iodata', 'saddle.periodic',
        'saddle.procrustes', 'saddle.procrustes.test', 'saddle.newopt'
    ],
    include_package_data=True,
    package_data={
        'saddle': [
            'data/*.json',
            'data/*.com',
        ],
        'saddle.periodic': ['data/*.csv'],
    },
    # package_data is only useful for bdist
    # add to MANIFEST.in works for both bdist and sdist
    data_files=[
        ('data', glob('data/*.*')),
        ('work', glob('work/*.*')),
        ('work/log', glob('work/log/.*')),
    ],
    install_requires=[
        'numpy', 'pytest'
    ], )
