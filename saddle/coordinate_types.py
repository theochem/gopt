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
"Coordinates types for represent internal coordinates."

import numpy as np

from scipy.misc import derivative
from typing import Tuple

from saddle.molmod import (bend_angle, bend_cos, bond_length, dihed_cos,
                           dihed_new_cross, dihed_new_dot)

__all__ = ('BondLength', 'BendAngle', 'BendCos', 'ConventionDihedral',
           'NewDihedralDot', 'NewDihedralCross')


class CoordinateTypes:
    """General internal coordinates class"""

    def __init__(self, atoms: "np.ndarray[int]",
                 coordinates: "np.ndarray[float]") -> None:
        self._coordinates = coordinates
        self._atoms = atoms
        self._value, self._d, self._dd = self._get_all()
        return None

    @property
    def value(self) -> float:
        return self._value

    def get_gradient_hessian(
            self) -> Tuple["np.ndarray[float]", "np.ndarray[float]"]:
        return self._d, self._dd

    def set_new_coordinates(self, new_coor: "np.ndarray[float]") -> None:
        self._coordinates = new_coor
        self._value, self._d, self._dd = self._get_all()
        return None

    @property
    def atoms(self) -> "np.ndarray[int]":
        return self._atoms

    def _get_all(self):
        raise NotImplementedError(
            "This method should be implemented in subclass")

    def get_cost(self, target):
        raise NotImplementedError(
            "This method should be implemented in subclass")


class BondLength(CoordinateTypes):
    """BondLength type internal coordinates class

    Properties
    ----------
    value : float
        The value of bond length
    atoms : np.ndarray(N,)
        The atoms consist of this internal coordinates
    info : string
        The string to describe the property and important information

    methods
    -------
    get_gradient_hessian()
        Obtain the transformation gradient and hessian of this internal
        coordinates
    set_new_coordinates(new_coor)
        Set the cartesian coordinates of this internal coodinates
    """

    def _get_all(self):
        return bond_length(self._coordinates, 2)

    @property
    def info(self) -> None:
        pass

    def __repr__(self) -> str:
        return "Bond-{}-({})".format(self.atoms, self.value)

    def get_cost(self, target):
        cost_v = (self.value - target) ** 2
        cost_d = 2 * (self.value - target)
        cost_dd = 2
        return cost_v, cost_d, cost_dd


class BendAngle(CoordinateTypes):
    """BendAngle type internal coordinates class

    Properties
    ----------
    value : float
        The value of bend_angle
    atoms : np.ndarray(N,)
        The atoms consist of this internal coordinates
    info : string
        The string to describe the property and important information

    methods
    -------
    get_gradient_hessian()
        Obtain the transformation gradient and hessian of this internal
        coordinates
    set_new_coordinates(new_coor)
        Set the cartesian coordinates of this internal coodinates
    """

    def _get_all(self):
        return bend_angle(self._coordinates, 2)

    @property
    def info(self) -> None:
        pass

    def __repr__(self) -> str:
        return "Angle-{}-({})".format(self.atoms, self.value)

    def get_cost(self, target):
        raise NotImplementedError('Not implemented yet')

class BendCos(CoordinateTypes):
    """BendCos type internal coordinates class

    Properties
    ----------
    value : float
        The value of cosine value of bend_angle
    atoms : np.ndarray(N,)
        The atoms consist of this internal coordinates
    info : string
        The string to describe the property and important information

    methods
    -------
    get_gradient_hessian()
        Obtain the transformation gradient and hessian of this internal
        coordinates
    set_new_coordinates(new_coor)
        Set the cartesian coordinates of this internal coodinates
    """

    def _get_all(self):
        return bend_cos(self._coordinates, 2)

    @property
    def info(self) -> None:
        pass

    def __repr__(self) -> str:
        return "Angle-{}-({})".format(self.atoms, self.value)

    def get_cost(self, target):
        cost_v = (self.value - target) ** 2
        cost_d = 2 * (self.value - target)
        cost_dd = 2
        return cost_v, cost_d, cost_dd

class ConventionDihedral(CoordinateTypes):
    """ConventionDihedral type internal coordinates class

    Properties
    ----------
    value : float
        The cosine value of dihedral
    atoms : np.ndarray(N,)
        The atoms consist of this internal coordinates
    info : string
        The string to describe the property and important information

    methods
    -------
    get_gradient_hessian()
        Obtain the transformation gradient and hessian of this internal
        coordinates
    set_new_coordinates(new_coor)
        Set the cartesian coordinates of this internal coodinates
    """

    def _get_all(self):
        return dihed_cos(self._coordinates, 2)

    @property
    def info(self) -> None:
        pass

    def __repr__(self) -> str:
        return "Dihed-{}-({})".format(self.atoms, self.value)



class NewDihedralDot(CoordinateTypes):  # need tests
    """NewDihedralDot type internal coordinates class

    Properties
    ----------
    value : float
        The value of new dihedral dot internal cooridnate
    atoms : np.ndarray(N,)
        The atoms consist of this internal coordinates
    info : string
        The string to describe the property and important information

    methods
    -------
    get_gradient_hessian()
        Obtain the transformation gradient and hessian of this internal
        coordinates
    set_new_coordinates(new_coor)
        Set the cartesian coordinates of this internal coodinates
    """

    def _get_all(self):
        return dihed_new_dot(self._coordinates, 2)

    @property
    def info(self) -> None:
        pass


class NewDihedralCross(CoordinateTypes):  # need tests
    """NewDihedralCross type internal coordinates class

    Properties
    ----------
    value : float
        The value of new dihedral cross internal cooridnate
    atoms : np.ndarray(N,)
        The atoms consist of this internal coordinates
    info : string
        The string to describe the property and important information

    methods
    -------
    get_gradient_hessian()
        Obtain the transformation gradient and hessian of this internal
        coordinates
    set_new_coordinates(new_coor)
        Set the cartesian coordinates of this internal coodinates
    """

    def _get_all(self):
        return dihed_new_cross(self._coordinates, 2)

    @property
    def info(self) -> None:
        pass

    def get_cost(self, target): # TODO: need to test
        sin_ang1 = np.sin(bend_angle(self._coordinates[:3])) ** 2
        sin_ang2 = np.sin(bend_angle(self._coordinates[1:])) ** 2
        sin_target = (1 - target**2) ** 0.5
        def cost_dihed(x):
            return sin_ang1 * sin_ang2 * ((x - target)**2 + ((1 - x**2)**0.5 - sin_target)**2)
        cost_v = cost_dihed(self.value)
        cost_d = derivative(cost_dihed, self.value, dx=1e-6, n=1)
        cost_dd = derivative(cost_dihed, self.value, dx=1e-6, n=2)
        return cost_v, cost_d, cost_dd
        # cost_dd =
