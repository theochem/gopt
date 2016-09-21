from __future__ import print_function, absolute_import
from saddle.cartesian import Cartesian
from saddle.errors import NotSetError, AtomsNumberError
from saddle.molmod import bond_length
from saddle.coordinate_types import Bond_Length
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
        # reorder the sequence of atoms indice
        atoms = self._atoms_sequence_reorder(atoms)
        rs = np.vstack(
            (self.coordinates[atoms[0]], self.coordinates[atoms[1]]))
        # gradient and hessian need to be set
        v, d, dd = bond_length(rs, deriv=2)
        new_ic_obj = Bond_Length(v, atoms)
        if self._repeat_check(new_ic_obj):  # repeat internal coordinates check
            self._ic.append(new_ic_obj)
            self._add_cc_to_ic_gradient(d, atoms) # update gradient
            self._add_cc_to_ic_hessian(dd, atoms) # update hessian
            self._add_connectivity(atoms)

    def add_angle(self, atom1, atom2, atom3):
        pass

    def add_dihedral(self, atom1, atom2, atom3):
        pass

    def set_targe_ic(self, new_ic):
        self._target_ic=new_ic

    def converge_to_target_ic(self):
        pass

    def _repeat_check(self, ic_obj):
        for ic in self.ic:
            if ic_obj.atoms == ic.atoms and type(ic_obj) == type(ic):
                return False
        else:
            return True

    def _add_connectivity(self, atoms):
        if len(atoms) != 2:
            raise AtomsNumberError, "The number of atoms is not correct"
        num1, num2 = atoms
        self._connectivity[num1, num2] = 1
        self._connectivity[num2, num1] = 1

    def _atoms_sequence_reorder(self, atoms):
        atoms=list(atoms)
        if len(atoms) == 2:
            if atoms[0] > atoms[1]:
                atoms[0], atoms[1]=atoms[1], atoms[0]
        elif len(atoms) == 3:
            if atoms[0] > atoms[2]:
                atoms[0], atoms[2]=atoms[2], atoms[0]
        elif len(atoms) == 4:
            if atoms[0] > atoms[3]:
                atoms[0], atoms[3]=atoms[3], atoms[0]
            if atoms[1] > atoms[2]:
                atoms[1], atoms[2]=atoms[2], atoms[1]
        else:
            raise AtomsNumberError, "The number of atoms is not correct"
        return tuple(atoms)

    def _add_cc_to_ic_gradient(self, deriv, atoms):  # need to be tested
        if self._cc_to_ic_gradient is None:
            self._cc_to_ic_gradient=np.zeros((0, 3 * len(self.numbers)))
        tmp_vector = np.zeros((1, 3 * len(self.numbers)))
        for i in range(len(atoms)):
            tmp_vector[0, 3 * atoms[i]: 3 * atoms[i] + 3] += deriv[i]
        self._cc_to_ic_gradient=np.vstack(
            (self._cc_to_ic_gradient, tmp_vector))

    def _add_cc_to_ic_hessian(self, deriv, atoms):  # need to be tested
        if self._cc_to_ic_hessian is None:
            self._cc_to_ic_hessian=np.zeros(
                (0, 3 * len(self.numbers), 3 * len(self.numbers)))
        tmp_vector=np.zeros((1, 3 * len(self.numbers), 3 * len(self.numbers)))
        for i in range(len(atoms)):
            for j in range(len(atoms)):
                tmp_vector[0, 3 * atoms[i]: 3 * atoms[i] + 3, 3 * \
                    atoms[j]: 3 * atoms[j] + 3] += deriv[i, :3, j]
        self._cc_to_ic_hessian=np.vstack((self._cc_to_ic_hessian, tmp_vector))

    @property
    def ic(self):
        return self._ic

    @property
    def ic_values(self):
        return [i.value for i in self._ic]

    @property
    def target_ic(self):
        return self._target_ic

    @property
    def connectivity(self):
        return self._connectivity

    def print_connectivity(self):
        format_func = "{:3}".format
        print ("--Connectivity Starts-- \n")
        for i in range(len(self.numbers)):
            print (" ".join(map(format_func, self.connectivity[i, :i + 1])))
        print ("\n--Connectivity Ends--")


# import horton as ht
# fn_xyz = ht.context.get_fn("test/water.xyz")
# mol = ht.IOData.from_file(fn_xyz)  # create a water molecule
# inter = Internal(mol.coordinates, mol.numbers, 0, 1)
# inter.add_bond(0,2)
# inter.add_bond(0,1)
# print (inter._cc_to_ic_hessian.shape)
