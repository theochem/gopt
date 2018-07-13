#!/usr/bin/env python3

from glob import glob

from setuptools import setup

setup(
    name="saddle",
    version="0.1",
    description="Geometry optimization program for chemical reaction",
    license='GPLv3',
    author='Derrick Yang',
    author_email='yxt1991@gmail.com',
    package_dir={'saddle': 'saddle'},
    packages=[
        'saddle',
        'saddle.periodic',
        'saddle.procrustes',
        'saddle.optimizer',
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
        ('work', glob('work/*.*')),
        ('work/log', glob('work/log/.*')),
    ],
)
