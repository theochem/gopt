from __future__ import print_function, absolute_import
import numpy as np
from saddle.abclass import CoordinateTypes
from saddle.errors import AtomsNumberError
from saddle.molmod import bond_length, bend_cos, dihed_cos, dihed_new_dot, dihed_new_cross

# class BondLength(CoordinateTypes):
#
#     def __init__(self, value, atoms):
#         self._value = value
#         # self._coordinates = coordinates
#         if len(atoms) != 2:
#             raise AtomsNumberError, "The number of atoms for this coordinate should be 2"
#         c_atoms = list(atoms)
#         if c_atoms[0] > c_atoms[1]:
#             c_atoms[0], c_atoms[1] = c_atoms[1], c_atoms[0]
#         self._atoms = tuple(c_atoms)
#
#     @property
#     def value(self):
#         return self._value
#
#     @property
#     def atoms(self):
#         return self._atoms
#
#     @property
#     def info(self):
#         pass

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

class NewConventionDot(CoordinateTypes): # to be fixed

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

class NewConventionCross(CoordinateTypes): # to be fixed

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

if __name__ =="__main__":
    a = Bond_Length(2.5,(3,2))
    assert(a.atoms == (2,3))
    b = Bend_Angle(1.5, (5,6,2))
    assert(b.atoms == (2,6,5))
