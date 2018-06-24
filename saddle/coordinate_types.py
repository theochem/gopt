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
from enum import Enum
from saddle.errors import NotSetError
from saddle.molmod import (bend_angle, bend_cos, bond_length, dihed_angle,
                           dihed_cos, dihed_new_cross, dihed_new_dot)
from typing import Tuple

__all__ = ('BondLength', 'BendAngle', 'BendCos', 'ConventionDihedral',
           'NewDihedralDot', 'NewDihedralCross')


class CoordinateTypes:
    """General internal coordinates class"""

    def __init__(self,
                 atoms: "np.ndarray[int]",
                 coordinates: "np.ndarray[float]",
                 *_,
                 weight=1) -> None:
        self._coordinates = coordinates
        self._atoms = atoms
        self._value, self._d, self._dd = self._get_all()
        self._weight = weight
        self._target = None
        return None

    @property
    def target(self):
        if self._target:
            return self._target
        raise NotSetError('target ic is not set')


    @target.setter
    def target(self, value):
        self._target = value

    @property
    def value(self) -> float:
        return self._value

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value

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

    # def get_cost(self):
    #     raise NotImplementedError(
    #         "This method should be implemented in subclass")
    @property
    def cost_v(self):
        raise NotImplementedError(
            "This method should be implemented in subclass")

    @property
    def cost_d(self):
        raise NotImplementedError(
            "This method should be implemented in subclass")

    @property
    def cost_dd(self):
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

    def __init__(self,
                 atoms: "np.ndarray[int]",
                 coordinates: "np.ndarray[float]",
                 *_,
                 ic_type=None,
                 weight=1) -> None:
        super().__init__(atoms, coordinates, weight=weight)
        self._ic_type = ic_type

    def _get_all(self):
        return bond_length(self._coordinates, 2)

    @property
    def info(self) -> None:
        pass

    def __repr__(self) -> str:
        if self.ic_type:
            return "Bond({})-{}-({})".format(
                BondLength._bond_type_dict[self.ic_type], self.atoms,
                self.value)
        else:
            return "Bond-{}-({})".format(self.atoms, self.value)

    @property
    def cost_v(self):
        return (self.value - self.target)**2 * self.weight

    @property
    def cost_d(self):
        return 2 * (self.value - self.target) * self.weight

    @property
    def cost_dd(self):
        return 2 * self.weight

    @property
    def ic_type(self):
        if self._ic_type:
            return self._ic_type
        else:
            pass

    _bond_type_dict = {
        0: 'NotSet',
        1: 'Covalent',
        2: 'Hydrogen',
        3: 'Inter-fragment',
        4: 'auxiliary',
        5: 'linear-chain'
    }


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

    @property
    def cost_v(self):
        return (np.cos(self.value) - np.cos(self.target))**2 * self.weight

    @property
    def cost_d(self):
        return -2 * (np.cos(self.value) - np.cos(self.target)) * np.sin(
            self.value) * self.weight

    @property
    def cost_dd(self):
        return 2 * (np.sin(self.value)**2 - np.cos(self.value)**2 +
                    np.cos(self.target) * np.cos(self.value)) * self.weight


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

    def get_cost(self):
        target = self.target
        cost_v = (self.value - target)**2
        cost_d = 2 * (self.value - target)
        cost_dd = 2
        return cost_v * self.weight, cost_d * self.weight, cost_dd * self.weight


class DihedralAngle(CoordinateTypes):
    """DihedralAngle type internal coordinates class

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

    @property
    def sub_weight(self):
        sin_ang1 = np.sin(bend_angle(self._coordinates[:3]))
        sin_ang2 = np.sin(bend_angle(self._coordinates[1:]))
        return sin_ang1**2 * sin_ang2**2

    @sub_weight.setter
    def sub_weight(self, value):
        self._weight = value

    def _get_all(self):
        return dihed_angle(self._coordinates, 2)

    @property
    def info(self) -> None:
        pass

    def __repr__(self) -> str:
        return "Dihed-{}-({})".format(self.atoms, self.value)

    @property
    def cost_v(self):
        return (2 - 2 * np.cos(self.value - self.target)
                ) * self.weight * self.sub_weight

    @property
    def cost_d(self):
        return 2 * np.sin(
            self.value - self.target) * self.weight * self.sub_weight

    @property
    def cost_dd(self):
        return 2 * np.cos(
            self.value - self.target) * self.weight * self.sub_weight


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

    def get_cost(self):  # TODO: need to test
        raise NotImplementedError
        # sin_ang1 = np.sin(bend_angle(self._coordinates[:3]))**2
        # sin_ang2 = np.sin(bend_angle(self._coordinates[1:]))**2
        # sin_target = (1 - target**2)**0.5
        # sin_value = (1 - self.value**2)**0.5
        # cost_v = -2 * sin_ang1 * sin_ang2 * (
        #     self.value * target + sin_value * sin_target)
        # cost_d = -2 * sin_ang1 * sin_ang2 * (
        #     target - (1 - self.value)**-0.5 * self.value * sin_target)
        # cost_dd = 2 * sin_ang1 * sin_ang2 * sin_target * (
        #     (1 - self.value**2)**-0.5 +
        #     (1 - self.value**2)**-1.5 * self.value**2)
        # return cost_v, cost_d, cost_dd


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

    def __repr__(self) -> str:
        return "Dihed_dot-{}-({})".format(self.atoms, self.value)

    @property
    def cost_v(self):
        return (self.value - self.target)**2 * self.weight

    @property
    def cost_d(self):
        return 2 * (self.value - self.target) * self.weight

    @property
    def cost_dd(self):
        return 2 * self.weight


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

    def __repr__(self) -> str:
        return "Dihed_cross-{}-({})".format(self.atoms, self.value)

    @property
    def cost_v(self):
        return (self.value - self.target)**2 * self.weight

    @property
    def cost_d(self):
        return 2 * (self.value - self.target) * self.weight

    @property
    def cost_dd(self):
        return 2 * self.weight
