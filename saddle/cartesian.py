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
"Cartesian coordinates implementation"

import numpy as np
import numpy.linalg as npl

from saddle.errors import AtomsNumberError, NotSetError
from saddle.fchk import FCHKFile
from saddle.gaussianwrapper import GaussianWrapper
from saddle.iodata import IOData

__all__ = ('Cartesian', )


class Cartesian:
    """ Cartesian Coordinate.

    Properties
    ----------
    numbers : np.ndarray(N)
        A list of atomic number for input coordinates
    spin : int
        Spin multiplicity of the molecule
    charge : int
        Charge of the input molecule
    energy : float
        Energy of given Cartesian coordinates system molecule
    energy_gradient : np.ndarray(N)
        Gradient of Energy that calculated under Cartesian coordinates
    energy_hessian : np.ndarray(N, N)
        Hessian of Energy that calculated under Cartesian coordinates
    coordinates : np.ndarray(K, 3)
        Cartesian information of input molecule
    natom : int
        Number of atoms in the system

    Classmethod
    -----------
    from_file(filename, charge=0, spin=1)
        Create cartesian instance from file

    Methods
    -------
    __init__(self, coordinates, numbers, charge, spin)
        Initializes molecule
    set_new_coordinates(new_coor)
        Set molecule with a set of coordinates
    energy_from_fchk(self, abs_path, gradient=True, hessian=True):
        Obtain energy and corresponding info from fchk file
    energy_calculation(**kwargs)
        Calculate system energy with different methods through software
        like gaussian
    distance(index1, index2)
        Calculate distance between two atoms with index1 and index2
    angle_cos(index1, index2, index3):
        Calculate radian of angle between atoms with index1, index2 and index3
    angle(index1, index2, index3)
        Calculate cosine of angle between atoms with index1, index2, and index3
    """

    def __init__(self,
                 coordinates: 'np.ndarray[float]',
                 numbers: 'np.ndarray[int]',
                 charge: int,
                 spin: int,
                 title: str = "untitled") -> None:
        self._coordinates = coordinates.copy()
        self._numbers = numbers.copy()
        self._charge = charge
        self._spin = spin
        self._title = title
        self._energy = None
        self._energy_gradient = None
        self._energy_hessian = None
        return None

    @classmethod
    def from_file(cls, filename: str, charge: int = 0,
                  spin: int = 1) -> 'Cartesian':
        """Create an Cartesian instance from file .xyz, .com,
        .gjf or .fchk

        Arguments
        ---------
        filename : str
            the path of the file
        charge : int, default is 0
            the charge of the given molecule(system)
        spin : int, dufault is 1
            the multiplicity of the given molecule(system)

        Return
        ------
        new Cartesian instance : Cartesian
        """
        mol = IOData.from_file(filename)
        return cls(mol.coordinates, mol.numbers, charge, spin)

    @property
    def energy_gradient(self) -> 'np.ndarray[float]':
        """Gradient of energy versus cartesian coordinates

        Returns
        -------
        energy_gradient : np.ndarray(3N,)
        """
        if self._energy_gradient is None:
            raise NotSetError(
                "The value 'energy_gradient' unset, do the calculation first")
        else:
            return self._energy_gradient

    @property
    def energy_hessian(self) -> 'np.ndarray[float]':
        """Hessian of energy versus internal coordinates

        Returns
        -------
        energy_hessian : np.ndarray(3N, 3N)
        """
        if self._energy_hessian is None:
            raise NotSetError(
                "The value 'energy_hessian' is None, do the calculation first")
        else:
            return self._energy_hessian

    @property
    def energy(self) -> float:
        """Energy of the system

        Returns
        -------
        energy : float
        """
        if self._energy is None:
            raise NotSetError(
                "The value 'energy' is None, do the calculation first")
        else:
            return self._energy

    def set_new_coordinates(self, new_coor: 'np.ndarray[float]') -> None:
        """Assign new cartesian coordinates to this molecule

        Arguments
        ---------
        new_coor : np.ndarray(N, 3)
            New cartesian coordinates of the system
        """
        if self._coordinates.shape != new_coor.shape:
            raise AtomsNumberError(
                "the dimentsion of coordinates are not the same")
        self._coordinates = new_coor.copy()
        self._reset_cartesian()
        return None

    def _reset_cartesian(self) -> None:
        """Reset the energy data including energy, gradient and hessian.
        """
        self._energy = None
        self._energy_gradient = None
        self._energy_hessian = None
        return None

    @property
    def numbers(self) -> 'np.ndarray[int]':
        """Atomic number of all the atoms in the system

        Returns
        -------
        numbers : an np.ndarray of atomic numbers, len(numbers) = N
        """
        return self._numbers

    @property
    def charge(self) -> int:
        """The charge of the system

        Returns
        -------
        charge : int
        """
        return self._charge

    @property
    def spin(self) -> int:
        """The spin multiplicity of the system

        Returns
        -------
        spin : int
        """
        return self._spin

    @property
    def coordinates(self) -> 'np.ndarray[float]':
        """Cartesian coordinates of every atoms

        Returns
        -------
        coordinates : np.ndarray(N, 3)
        """
        return self._coordinates

    @property
    def natom(self) -> int:
        """number of atoms of given molecule

        Returns
        -------
        natom : int
        """
        return len(self.numbers)

    def energy_from_fchk(self,
                         abs_path: str,
                         *_,
                         gradient: bool = True,
                         hessian: bool = True) -> None:
        """Obtain energy and relative information from FCHK file.

        Arguments
        ---------
        abs_path : str
            Absolute path of fchk file in filesystem
        gradient : bool
            True if want to obtain gradient information, otherwise False.
            Default value is True
        hessian : bool
            True if want to obtain hessian information, otherwise False.
            Default value is True
        """
        fchk_file = FCHKFile(filename=abs_path)
        self.set_new_coordinates(fchk_file.get_coordinates().reshape(-1, 3))
        self._energy = fchk_file.get_energy()
        if gradient:
            self._energy_gradient = fchk_file.get_gradient()
        if hessian:
            self._energy_hessian = fchk_file.get_hessian()
        return None

    def energy_calculation(self, *_, method: str = 'g09',
                           title: str) -> None:  # need test
        """Conduct calculation with designated method.

        Keywords Arguments
        ------------------
        method : str, default is 'g09'
            name of the program(method) used to calculate energy and other
            property
        """
        # method = kwargs.pop('method', 'g09')  # get calculation method arg
        title = self._title
        if method == "g09":
            obj = GaussianWrapper(self, title)
            coor, ener, grad, hess = obj.run_gaussian_and_get_result(
                self.charge,
                self.spin,
                energy=True,
                gradient=True,
                hessian=True)
            self.set_new_coordinates(coor.reshape(-1, 3))
            self._energy = ener
            self._energy_gradient = grad
            self._energy_hessian = hess
        return None

        # set new coordinates after rotation
        # set self._energy
        # set self._energy_gradient
        # sel self._energy_hessian

    def distance(self, index1: int, index2: int) -> float:
        """Calculate the distance between two atoms

        Arguments
        ---------
        index1 : int
            The index of the first atom for calculating the distance
        index2 : int
            The index of the second atom for calculating the distance

        Return
        ------
        distance : float
            the distance between two atoms
        """
        coord1 = self.coordinates[index1]
        coord2 = self.coordinates[index2]
        diff = coord1 - coord2
        distance = npl.norm(diff)
        return distance

    def angle_cos(self, index1: int, index2: int, index3: int) -> float:
        """Calculate cosine of the angle consist of vector (index1 - index2)
        and vector (index3 - index2)

        Arguments
        ---------
        index1 : int
            The index of the first atom for angle
        index2 : int
            The index of the second atom for angle
        index3 : int
            The index of the third atom for angle

        Returns
        -------
        cos_angle : float
            cosine value of angle
        """
        coord1 = self.coordinates[index1]
        coord2 = self.coordinates[index2]
        coord3 = self.coordinates[index3]
        diff_1 = coord2 - coord1
        diff_2 = coord2 - coord3
        cos_angle = np.dot(diff_1, diff_2) / \
            (npl.norm(diff_1) * npl.norm(diff_2))
        return cos_angle

    def angle(self, index1: int, index2: int, index3: int) -> float:
        """Calculate radian of the angle consist of vector (index1 - index2)
        and vector (index3 - index2)

        Arguments
        ---------
        index1 : int
            The index of the first atom for angle
        index2 : int
            The index of the second atom for angle
        index3 : int
            The index of the third atom for angle

        Returns
        -------
        cos_angle : float
            radian value of angle
        """
        cos_value = self.angle_cos(index1, index2, index3)
        return np.arccos(cos_value)
