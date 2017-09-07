#!/usr/bin/env python

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
        'saddle', 'saddle.test', 'saddle.iodata', 'saddle.periodic',
        'saddle.procrustes', 'saddle.newopt'
    ],
    package_data={
        'saddle': [
            'data/*.json', 'data/*.com', 'data/*.xyz', 'data/*.fchk',
            'work/log/.gitkeep'
        ],
        'saddle.periodic': ['data/*.csv'],
    },
    install_requires=[
        'numpy',
    ], )
