from __future__ import absolute_import, print_function

import numpy as np

from saddle.abclass import CoordinateTypes
from saddle.errors import AtomsNumberError
from saddle.molmod import (bend_angle, bend_cos, bond_length, dihed_cos,
                           dihed_new_cross, dihed_new_dot)


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

    def __init__(self, atoms, coordinates):
        self._coordinates = coordinates
        self._atoms = atoms
        self._value, self._d, self._dd = bond_length(self._coordinates, 2)

    @property
    def value(self):
        return self._value

    def get_gradient_hessian(self):
        return self._d, self._dd

    def set_new_coordinates(self, new_coor):
        self._coordinates = new_coor
        self._value, self._d, self._dd = bond_length(self._coordinates, 2)

    @property
    def atoms(self):
        return self._atoms

    @property
    def info(self):
        pass

    def __repr__(self):
        return "Bond-{}-({})".format(self.atoms, self.value)


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

    def __init__(self, atoms, coordinates):
        self._coordinates = coordinates
        self._atoms = atoms
        self._value, self._d, self._dd = bend_angle(self._coordinates, 2)

    @property
    def value(self):
        return self._value

    def get_gradient_hessian(self):
        return self._d, self._dd

    def set_new_coordinates(self, new_coor):
        self._coordinates = new_coor
        self._value, self._d, self._dd = bend_angle(self._coordinates, 2)

    @property
    def atoms(self):
        return self._atoms

    @property
    def info(self):
        pass

    def __repr__(self):
        return "Angle-{}-({})".format(self.atoms, self.value)


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

    def __init__(self, atoms, coordinates):
        self._coordinates = coordinates
        self._atoms = atoms
        self._value, self._d, self._dd = bend_cos(self._coordinates, 2)

    @property
    def value(self):
        return self._value

    def get_gradient_hessian(self):
        return self._d, self._dd

    def set_new_coordinates(self, new_coor):
        self._coordinates = new_coor
        self._value, self._d, self._dd = bend_cos(self._coordinates, 2)

    @property
    def atoms(self):
        return self._atoms

    @property
    def info(self):
        pass

    def __repr__(self):
        return "Angle-{}-({})".format(self.atoms, self.value)


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

    def __init__(self, atoms, coordinates):
        self._coordinates = coordinates
        self._atoms = atoms
        self._value, self._d, self._dd = dihed_cos(self._coordinates, 2)

    @property
    def value(self):
        return self._value

    def get_gradient_hessian(self):
        return self._d, self._dd

    def set_new_coordinates(self, new_coor):
        self._coordinates = new_coor
        self._value, self._d, self._dd = dihed_cos(self._coordinates, 2)

    @property
    def atoms(self):
        return self._atoms

    @property
    def info(self):
        pass

    def __repr__(self):
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

    def __init__(self, atoms, coordinates):
        self._coordinates = coordinates
        self._atoms = atoms
        self._value, self._d, self._dd = dihed_new_dot(self._coordinates, 2)

    def get_gradient_hessian(self):
        return self._d, self._dd

    def set_new_coordinates(self, new_coor):
        self._coordinates = new_coor
        self._value, self._d, self._dd = dihed_new_dot(self._coordinates, 2)

    @property
    def value(self):
        return self._value

    @property
    def atoms(self):
        return self._atoms

    @property
    def info(self):
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

    def __init__(self, atoms, coordinates):
        self._coordinates = coordinates
        self._atoms = atoms
        self._value, self._d, self._dd = dihed_new_cross(self._coordinates, 2)

    def get_gradient_hessian(self):
        return self._d, self._dd

    def set_new_coordinates(self, new_coor):
        self._coordinates = new_coor
        self._value, self._d, self._dd = dihed_new_cross(self._coordinates, 2)

    @property
    def value(self):
        return self._value

    @property
    def atoms(self):
        return self._atoms

    @property
    def info(self):
        pass
