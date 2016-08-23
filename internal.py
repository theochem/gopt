from __future__ import print_function, absolute_import
from cartesian import Cartesian
from errors import NotSetError
import numpy as np

class Internal(Cartesian):

    def __init__(self, coordinates, numbers, charge, spin):
        super(Internal, self).__init__(coordinates, numbers)
        self._ic = np.array([]) # type np.array([float 64])
        self._connectivity = np.zeros([len(self.numbers), len(self.numbers)])
        self._target_ic = None
        self._ic_to_cc_gradient = None
        self._ic_to_cc_hessiant = None

    def add_bond(self, atom1, atom2):
        pass

    def add_angle(self, atom1, atom2, atom3):
        pass

    def add_dihedral(self, atom1, atom2, atom3):
        pass

    def set_targe_ic(self, new_ic):
        self._target_ic = new_ic

    def converge_to_target_ic(self):
        pass


a = Internal(1,2)
print(a.coordinates)
print(a.energy_gradient)
# print(a._numbers)
