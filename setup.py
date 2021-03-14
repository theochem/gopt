#!/usr/bin/env python3
"""Setup file for installing GOpt."""
import gopt

from setuptools import find_packages, setup

"""Setup installation dependencies."""
setup(
    name="gopt",
    version=gopt.__version__,
    description="Geometry optimization program for chemical reaction",
    license=gopt.__license__,
    author=gopt.__author__,
    author_email="yxt1991@gmail.com",
    package_dir={"gopt": "gopt"},
    packages=find_packages(exclude=["*.test", "*.test.*", "test.*", "test"]),
    include_package_data=True,
    package_data={
        "gopt": ["data/*.json", "data/*.com", "work/log/.gitkeep"],
        "gopt.periodic": ["data/*.csv"],
    },
    install_requires=[
        "numpy>=1.16",
        "pytest>=2.6",
        #        "scipy>=1.2",
    ]
    # package_data is only useful for bdist
    # add to MANIFEST.in works for both bdist and sdist
)
