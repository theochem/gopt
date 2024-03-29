#!/usr/bin/env python3
"""Setup file for installing GOpt."""
import saddle

from setuptools import find_packages, setup

"""Setup installation dependencies."""
setup(
    name="saddle",
    version=saddle.__version__,
    description="Geometry optimization program for chemical reaction",
    license=saddle.__license__,
    author=saddle.__author__,
    author_email="yxt1991@gmail.com",
    package_dir={"saddle": "saddle"},
    packages=find_packages(exclude=["*.test", "*.test.*", "test.*", "test"]),
    include_package_data=True,
    package_data={
        "saddle": ["data/*.json", "data/*.com", "work/log/.gitkeep"],
        "saddle.periodic": ["data/*.csv"],
    },
    install_requires=[
        "numpy>=1.16",
        "pytest>=2.6",
        #        "scipy>=1.2",
        "importlib_resources",
    ]
    # package_data is only useful for bdist
    # add to MANIFEST.in works for both bdist and sdist
)
