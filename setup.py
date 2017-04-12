#!/usr/bin/env python

from setuptools import setup

setup(
    name="saddle",
    version="0.1",
    description="Geometry reformative advanced platform for education",
    author='Derrick Yang',
    author_email='yxt1991@gmail.com',
    package_dir={
        'saddle': 'saddle'
    },
    packages=['saddle', 'saddle.test', 'saddle.iodata', 'saddle.periodic',
              'saddle.data', 'saddle.newopt'],
    package_data={
        'saddle': ['data/*.xyz', 'data/*.fchk'],
        'saddle.periodic': ['data/*.csv'],
    },
    install_requires=['numpy', ], )
