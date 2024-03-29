# -*- coding: utf-8 -*-
# PyGopt: Python Geometry Optimization.
# Copyright (C) 2011-2018 The HORTON/PyGopt Development Team
#
# This file is part of PyGopt.
#
# PyGopt is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# PyGopt is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"""wrap over Gaussian to run gaussian calculation."""

import os
from pathlib import Path
from string import Template

from importlib_resources import read_text

import numpy as np

from saddle.conf import WORK_DIR
from saddle.fchk import FCHKFile
from saddle.periodic.periodic import angstrom, periodic

__all__ = ("GaussianWrapper",)


class GaussianWrapper(object):
    """Gaussian wrapper class."""

    counter = 0

    template = Template(read_text("saddle.data", "single_hf_template.com"))

    def __init__(self, molecule, title):
        """Initialize Gaussian Wrapper class.

        Parameters
        ----------
        molecule : Molecule
            Molecule instance with coordinates and numbers
        title : str
            molecule title
        """
        self.molecule = molecule
        self.title = title

    def run_gaussian_and_get_result(
        self,
        charge,
        multi,
        *_,
        coordinates=True,
        energy=True,
        gradient=False,
        hessian=False
    ):
        """Run gaussian calculation and get the result from .fchk file.

        Parameters
        ----------
        charge : int
            Charge of the molecule
        multi : int
            Multiplicity of the molecule
        coordinates : bool, kwarg, optional
            get coordinates from the result
        energy : bool, kwarg, optional
            get energy from the result
        gradient : bool, kwarg, optional
            get eneryg gradient from the result
        hessian : bool, kwarg, optional
            get energy hessian from the result
        """
        freq = ""
        if gradient or hessian:
            freq = "freq"
        # TODO: if gradient, use Force, if hessian, use FREQ
        filename = self._create_input_file(charge, multi, freq=freq)
        # print "gausian is going to run \n{} \n{} \n{}".format(charge, multi,
        #   self.molecule.ic)
        fchk_file = self._run_gaussian(filename)
        assert isinstance(
            fchk_file, FCHKFile
        ), "Gaussian calculation didn't run properly"
        result = [None] * 4
        if coordinates:
            result[0] = fchk_file.get_coordinates()
        if energy:
            result[1] = fchk_file.get_energy()
        if gradient:
            result[2] = fchk_file.get_gradient()
        if hessian:
            result[3] = fchk_file.get_hessian()
        return result

    def create_gauss_input(
        self, charge, multi, freq="freq", spe_title="", path="", postfix=".com"
    ):
        """Create a gaussian style input file.

        Parameters
        ----------
        charge : int
            molecular charge
        multi : int
            molecular multiplicity
        freq : str, optional
            frequency calculation keyward
        spe_title : str, optional
            file name
        path : str, optional
            path of the file to be created
        postfix : str, optional
            poxis of input file, .com or .gjf
        """
        assert isinstance(path, str) or isinstance(path, Path)
        assert isinstance(spe_title, str)
        atoms = ""
        for i in range(len(self.molecule.numbers)):
            x, y, z = self.molecule.coordinates[i] / angstrom
            atoms += "%2s % 10.5f % 10.5f % 10.5f \n" % (
                periodic[self.molecule.numbers[i]].symbol,
                x,
                y,
                z,
            )
        if spe_title:
            filename = spe_title
        elif self.title:
            filename = self.title
        else:
            raise ValueError("file name is not specified")
        if path:
            path = os.path.join(path, filename + postfix)
        else:
            path = os.path.join(WORK_DIR, filename + postfix)
        with open(path, "w") as f:
            f.write(
                self.template.substitute(
                    charge=charge, freq=freq, multi=multi, atoms=atoms, title=filename
                )
            )
        GaussianWrapper.counter += 1

    def _create_input_file(self, charge, multi, freq="freq"):
        """Create input file for gaussian."""
        filename = "{0}_{1}".format(self.title, self.counter)
        self.create_gauss_input(charge, multi, freq=freq, spe_title=filename)
        return filename

    def _run_gaussian(self, filename, fchk=True, command_bin="g09"):
        """Run gaussian calculation and format the output file to .fchk."""
        fchk_ob = None
        path = WORK_DIR
        os.chdir(path)
        os.system("{0} {1}.com".format(command_bin, filename))
        if fchk:
            logname = "{0}.log".format(filename)
            if os.path.isfile(os.path.join(path, logname)) and self._log_finish_test(
                os.path.join(path, logname)
            ):
                os.system(
                    "formchk {0}.chk {0}.fchk".format(os.path.join(path, filename))
                )
                fchk_ob = FCHKFile("{0}.fchk".format(os.path.join(path, filename)))
        # os.chdir(os.path.join(self.pwd, '..'))
        # print("change_back", self.pwd)
        return fchk_ob

    def _log_finish_test(self, logname):
        """Check whether the calculation run properly."""
        flag = False
        with open(logname) as f:
            for line in f:
                if "Normal termination" in line:
                    flag = True
        return flag


if __name__ == "__main__":
    from collections import namedtuple

    molecule = namedtuple("molecule", "numbers, coordinates")
    aa = molecule([1, 3], np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    a = GaussianWrapper(aa, "text_wrapper")
    print(a.template)
    a._create_input_file(0, 2)
    a._create_input_file(0, 2, "freq")
