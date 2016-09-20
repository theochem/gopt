from __future__ import print_function, absolute_import
from .cartesian import Cartesian
from .errors import NotSetError, AtomsNumberError
from .molmod import bond_length
from .coordinate_types import Bond_Length
import numpy as np


class Internal(Cartesian):

    def __init__(self, coordinates, numbers, charge, spin):
        super(Internal, self).__init__(coordinates, numbers, charge, spin)
        self._ic = []  # type np.array([float 64])
        # 1 is connected, 0 is not, -1 is itself
        self._connectivity = np.diag([-1] * len(self.numbers))
        self._target_ic = None
        self._cc_to_ic_gradient = None
        self._cc_to_ic_hessian = None

    def add_bond(self, atom1, atom2):
        atoms = (atom1, atom2)
        atoms = self._atoms_sequence_reorder(atoms) # reorder the sequence of atoms indice
        rs = np.vstack((self.coordinates[atoms[0]], self.coordinates[atoms[1]]))
        # gradient and hessian need to be set
        v, d, dd = bond_length(rs, deriv=2)
        new_ic_obj = Bond_Length(v, atoms))
        if self._repeat_check(new_ic_obj):  # repeat internal coordinates check
            self._ic.append(new_ic_obj)
            # gradient
            # hessian

    def add_angle(self, atom1, atom2, atom3):
        pass

    def add_dihedral(self, atom1, atom2, atom3):
        pass

    def set_targe_ic(self, new_ic):
        self._target_ic = new_ic

    def converge_to_target_ic(self):
        pass

    def _repeat_check(self, ic_obj):
        for ic in range(self.ic):
            if ic_obj.atoms == ic.atoms and type(ic_obj) == type(ic):
                return False
        else:
            return True

    def _atoms_sequence_reorder(self, atoms):
        atoms = list(atoms)
        if len(atoms) == 2:
            if atoms[0] > atoms[1]:
                atoms[0], atoms[1] = atoms[1], atoms[0]
        elif len(atoms) == 3:
            if atoms[0] > atoms[2]:
                atoms[0], atoms[2] = atoms[2], atoms[0]
        elif len(atoms) == 4:
            if atoms[0] > atoms[3]:
                atoms[0], atoms[3] = atoms[3], atoms[0]
            if atoms[1] > atoms[2]:
                atoms[1], atoms[2] = atoms[2], atoms[1]
        else:
            raise AtomsNumberError, "The number of atoms is not correct"
        return tuple(atoms)

    def _add_cc_to_ic_gradient(self, deriv, atoms): # need to be tested
        if self._cc_to_ic_gradient == None:
            self._cc_to_ic_gradient = np.zeros((0, 3 * len(self.numbers)))
        tmp_vector = np.zeros((1, 3 * len(self.numbers)))
        for i in range(len(atoms)):
            tmp_vector[1, 3 * atoms[i]: 3 * atoms[i] + 3] += deriv[i]
        self._cc_to_ic_gradient = np.vstack((self._cc_to_ic_gradient, tmp_vector))

    def _add_cc_to_ic_hessian(self, deriv, atoms): # need to be tested
        if self._cc_to_ic_hessian == None:
            self._cc_to_ic_hessian = np.zeros((0, 3 * len(self.numbers), 3 * len(self.numbers)))
        tmp_vector = np.zeros((3 * len(self.numbers), 3 * len(self.numbers)))
        for i in range(len(atoms)):
            tmp_vector[3 * atoms[i]: 3 * atoms[i] + 3, 3 * atoms[i] : 3 * atoms[i] + 3] += deriv[i][:3][i]
        self._cc_to_ic_hessian = np.vstack((self._cc_to_ic_hessian, tmp_vector))



    @property
    def ic(self):
        return self._ic

    @property
    def ic_values(self):
        return [i.values for i in self._ic]

    @property
    def target_ic(self):
        return self._target_ic

    @property
    def connectivity(self):
        return self._connectivity

    def print_connectivity(self):
        for i in range(len(self.numbers)):
            print("".join(map(str, self.connectivity[i, :i + 1])))


aaaa = Internal(1, [2], 3, 1)
print(aaaa.coordinates)
# print(a.energy_gradient)
# print(a._numbers)
