from __future__ import absolute_import, print_function

import numpy as np

from saddle.abclass import CoordinateTypes
from saddle.errors import AtomsNumberError
from saddle.molmod import (bend_cos, bond_length, dihed_cos, dihed_new_cross,
                           dihed_new_dot)


class BondLength(CoordinateTypes):
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


class NewConventionDot(CoordinateTypes):  # to be fixed
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


class NewConventionCross(CoordinateTypes):  # to be fixed
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


if __name__ == "__main__":
    a = BondLength(2.5, (3, 2))
    assert (a.atoms == (2, 3))
    b = BendAngle(1.5, (5, 6, 2))
    assert (b.atoms == (2, 6, 5))
