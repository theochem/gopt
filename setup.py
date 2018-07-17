#!/usr/bin/env python3

from setuptools import setup
import saddle

setup(
    name="saddle",
    version=saddle.__version__,
    description="Geometry optimization program for chemical reaction",
    license=saddle.__license__,
    author=saddle.__author__,
    author_email='yxt1991@gmail.com',
    package_dir={'saddle': 'saddle'},
    packages=[
        'saddle', 'saddle.periodic', 'saddle.procrustes', 'saddle.optimizer',
        'saddle.data', 'saddle.periodic.data'
    ],
    include_package_data=True,
    package_data={
        'saddle': ['data/*.json', 'data/*.com', 'work/log/.gitkeep'],
        'saddle.periodic': ['data/*.csv'],
    },
    # package_data is only useful for bdist
    # add to MANIFEST.in works for both bdist and sdist
)
