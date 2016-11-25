from __future__ import absolute_import, print_function

import numpy as np

from horton import periodic
from saddle.cartesian import Cartesian
from saddle.coordinate_types import BendAngle, BondLength, ConventionDihedral
from saddle.cost_functions import direct_square
from saddle.errors import (AtomsIndexError, AtomsNumberError, NotConvergeError,
                           NotSetError)
from saddle.opt import GeoOptimizer, Point


class Internal(Cartesian):
    """ Cartesian Coordinate

    Properties
    ----------
    numbers : np.ndarray(K)
        A list of atomic number for input coordinates
    spin : int
        Spin multiplicity of the molecule
    charge : int
        Charge of the input molecule
    energy : float
        Energy of given Cartesian coordinates system molecule
    energy_gradient : np.ndarray(K)
        Gradient of Energy that calculated through certain method
    energy_hessian : np.ndarray(K, K)
        Hessian of Energy that calculated through cartain method
    coordinates : np.ndarray(K, 3)
        Cartesian information of input molecule
    cost_value_in_cc : tuple(float, np.ndarray(K), np.ndarray(K, K))
        Return the cost function value, 1st, and 2nd
        derivative verse cartesian coordinates
    ic : list[CoordinateTypes]
        A list of CoordinateTypes instance to represent
        internal coordinates information
    ic_values : list[float]
        A list of internal coordinates values
    target_ic : list[float]
        A list of target internal coordinates
    connectivity : np.ndarray(K, K)
        A square matrix represents the connectivity of molecule
        internal coordinates

    Methods
    -------
    __init__(self, coordinates, numbers, charge, spin)
        Initializes molecule
    set_new_coordinates(new_coor)
        Set molecule with a set of coordinates
    energy_calculation(**kwargs)
        Calculate system energy with different methods through
        software like gaussian
    distance(index1, index2)
        Calculate distance between two atoms with index1 and index2
    angle(index1, index2, index3)
        Calculate angle between atoms with index1, index2, and index3
    add_bond(atom1, atom2)
        Add a bond between atom1 and atom2
    add_angle_cos(atom1, atom2, atom3)
        Add a cos of a angle consist of atom1, atom2, and atom3
    add_dihedral(atom1, atom2, atom3, atom4)
        Add a dihedral of plain consists of atom1, atom2, and atom3
        and the other one consist of atom2, atom3, and atom4
    set_target_ic(new_ic)
        Set a target internal coordinates to transform into
    converge_to_target_ic(iteration=100, copy=True)
        Implement optimization process to transform geometry to
        target internal coordinates
    print_connectivity()
        print connectivity matrix information on the screen
    """

    def __init__(self, coordinates, numbers, charge, spin):
        super(Internal, self).__init__(coordinates, numbers, charge, spin)
        self._ic = []  # type np.array([float 64])
        # 1 is connected, 0 is not, -1 is itself
        self._connectivity = np.diag([-1] * len(self.numbers))
        self._target_ic = None
        self._cc_to_ic_gradient = None
        self._cc_to_ic_hessian = None
        self._internal_gradient = None
        self._internal_hessian = None
        # self._tilt_internal_hessian = None

    def add_bond(self, atom1, atom2):  # tested
        if atom1 == atom2:
            raise AtomsIndexError("The two indece are the same")
        atoms = (atom1, atom2)
        # reorder the sequence of atoms indice
        atoms = self._atoms_sequence_reorder(atoms)
        rs = self.coordinates[np.array(atoms)]
        new_ic_obj = BondLength(atoms, rs)
        d, dd = new_ic_obj.get_gradient_hessian()
        # gradient and hessian need to be set
        if self._repeat_check(new_ic_obj):  # repeat internal coordinates check
            self._add_new_internal_coordinate(new_ic_obj, d, dd, atoms)
            # after adding a bond, change the connectivity of atoms pair to 1
            self._add_connectivity(atoms)

    def add_angle_cos(self, atom1, atom2, atom3):  # tested
        if atom1 == atom3:
            raise AtomsIndexError("The two indece are the same")
        atoms = (atom1, atom2, atom3)
        atoms = self._atoms_sequence_reorder(atoms)
        rs = self.coordinates[np.array(atoms)]
        new_ic_obj = BendAngle(atoms, rs)
        d, dd = new_ic_obj.get_gradient_hessian()
        # check if the angle is formed by two connected bonds
        if self._check_connectivity(atom1, atom2) and self._check_connectivity(
                atom2, atom3):
            if self._repeat_check(new_ic_obj):
                self._add_new_internal_coordinate(new_ic_obj, d, dd, atoms)

    def add_dihedral(self, atom1, atom2, atom3, atom4):  # tested
        if atom1 == atom4 or atom2 == atom3:
            raise AtomsIndexError("The two indece are the same")
        atoms = (atom1, atom2, atom3, atom4)
        atoms = self._atoms_sequence_reorder(atoms)
        rs = self.coordinates[np.array(atoms)]
        new_ic_obj = ConventionDihedral(atoms, rs)
        d, dd = new_ic_obj.get_gradient_hessian()
        if (self._check_connectivity(atom2, atom3) and
            (self._check_connectivity(atom1, atom2) or
             self._check_connectivity(atom1, atom3)) and
            (self._check_connectivity(atom4, atom3) or
             self._check_connectivity(atom4, atom2))):
            if self._repeat_check(new_ic_obj):
                self._add_new_internal_coordinate(new_ic_obj, d, dd, atoms)

    def set_target_ic(self, new_ic):
        if len(new_ic) != len(self.ic):
            raise AtomsNumberError("The ic is not in the same shape")
        self._target_ic = np.array(new_ic).copy()

    def set_new_coordinates(self, new_coor):  # to be tested
        super(Internal, self).set_new_coordinates(new_coor)
        self._regenerate_ic()

    def swap_internal_coordinates(self, index_1, index_2):
        self._ic[index_1], self._ic[index_2] = self._ic[index_2], self._ic[
            index_1]
        self._regenerate_ic()

    def converge_to_target_ic(self, iteration=100):  # to be test
        optimizer = GeoOptimizer()
        init_point = self._create_geo_point()
        optimizer.add_new(init_point)
        for i in range(iteration):
            optimizer.tweak_hessian(optimizer.newest)
            step = optimizer.trust_radius_step(optimizer.newest)
            new_coor = self.coordinates + step.reshape(-1, 3)
            self.set_new_coordinates(new_coor)
            new_point = self._create_geo_point()
            optimizer.add_new(new_point)
            if optimizer.converge(optimizer.newest):
                print("finished")
                return
            optimizer.update_trust_radius(optimizer.newest)
        raise NotConvergeError("The optimization failed to converge")

    def connected_indices(self, index):
        connection = self.connectivity[index]
        connected_index = np.where(connection > 0)[0]
        return connected_index

    def energy_from_fchk(self, abs_path, gradient=True, hessian=True):
        super(Internal, self).energy_from_fchk(abs_path, gradient, hessian)
        self._energy_hessian_transformation()

    def energy_calculation(self, **kwargs):
        super(Internal, self).energy_calculation(**kwargs)
        self._energy_hessian_transformation()
        # h_q = (B^T)^+ \cdot (H_x - K) \cdot B^+

    @property
    def cost_value_in_cc(self):
        v, d, dd = self._cost_value()
        x_d, x_dd = self._ic_gradient_hessian_transform_to_cc(d, dd)
        return v, x_d, x_dd

    @property
    def ic(self):
        return self._ic

    @property
    def ic_values(self):
        value = [i.value for i in self._ic]
        return np.array(value)

    @property
    def target_ic(self):
        return self._target_ic

    @property
    def connectivity(self):
        return self._connectivity

    @property
    def b_matrix(self):
        return self._cc_to_ic_gradient

    @property
    def internal_gradient(self):
        return self._internal_gradient

    def print_connectivity(self):
        format_func = "{:3}".format
        print("--Connectivity Starts-- \n")
        for i in range(len(self.numbers)):
            print(" ".join(map(format_func, self.connectivity[i, :i + 1])))
            print("\n--Connectivity Ends--")

    def auto_select_ic(self, dihed_special=False):
        self._auto_select_bond()
        self._auto_select_angle()
        self._auto_select_dihed_normal()
        self._auto_select_dihed_improper()

    def _auto_select_bond(self):
        halidish_atom = set([7, 8, 9, 15, 16, 17])
        for index_i in range(len(self.numbers)):
            for index_j in range(index_i + 1, len(self.numbers)):
                atom_num1 = self.numbers[index_i]
                atom_num2 = self.numbers[index_j]
                distance = self.distance(index_i, index_j)
                radius_sum = periodic[atom_num1].cov_radius + periodic[
                    atom_num2].cov_radius
                if distance < 1.3 * radius_sum:
                    self.add_bond(index_i, index_j)
                    if (min(atom_num1, atom_num2) == 1 and
                            max(atom_num1, atom_num2) in halidish_atom):
                        if atom_num1 == 1:
                            h_index = index_i
                            halo_index = index_j
                        else:
                            h_index = index_j
                            halo_index = index_i
                        for index_k in range(index_j + 1, len(self._numbers)):
                            atom_num3 = self.numbers[index_k]
                            if atom_num3 in halidish_atom:
                                dis = self.distance(h_index, index_k)
                                angle = self.angle(halo_index, h_index,
                                                   index_k)
                                thresh_sum = periodic[self._numbers[
                                    h_index]].vdw_radius + \
                                    periodic[self._numbers[index_k]].vdw_radius
                                if dis <= 0.9 * thresh_sum and angle >= 1.5708:
                                    self.add_bond(h_index, index_k)

    def _auto_select_angle(self):
        for center_index in range(len(self.numbers)):
            connected = self.connected_indices(center_index)
            if len(connected) >= 2:
                for edge_1 in range(len(connected)):
                    for edge_2 in range(edge_1 + 1, len(connected)):
                        self.add_angle_cos(connected[edge_1], center_index,
                                           connected[edge_2])

    def _auto_select_dihed_normal(self):
        for center_ind_1 in range(len(self.numbers)):
            connected = self.connected_indices(center_ind_1)
            if len(connected) >= 2:
                for center_ind_2 in connected:
                    sum_cnct = np.sum(self.connectivity, axis=0)
                    sum_select_cnct = sum_cnct[connected]
                    sorted_index = sum_select_cnct.argsort()[::-1]
                    side_1 = connected[sorted_index[0]]
                    if connected[sorted_index[0]] == center_ind_2:
                        side_1 = connected[sorted_index[1]]
                    connected_to_index_2 = self.connected_indices(center_ind_2)
                    for side_2 in connected_to_index_2:
                        if side_2 not in (center_ind_1, center_ind_2, side_1):
                            self.add_dihedral(side_1, center_ind_1,
                                              center_ind_2, side_2)

    def _auto_select_dihed_improper(self):
        connect_sum = np.sum(self.connectivity, axis=0)
        for center_ind in range(len(connect_sum)):
            if connect_sum[center_ind] >= 3:
                cnct_atoms = self.connected_indices(center_ind)
                cnct_total = len(cnct_atoms)
                for i in range(cnct_total):
                    for j in range(i + 1, cnct_total):
                        for k in range(j + 1, cnct_total):
                            ind_i, ind_j, ind_k = cnct_atoms[[i, j, k]]
                            ang1_r = self.angle(ind_i, center_ind,
                                                          ind_j)
                            ang2_r = self.angle(ind_i, center_ind,
                                                          ind_k)
                            ang3_r = self.angle(ind_j, center_ind,
                                                          ind_k)
                            sum_r = ang1_r + ang2_r + ang3_r
                            if sum_r >= 6.02139:
                                self.add_dihedral(ind_i, center_ind, ind_j,
                                                  ind_k)

    def _energy_hessian_transformation(self):
        self._internal_gradient = np.dot(
            np.linalg.pinv(self._cc_to_ic_gradient.T), self._energy_gradient)
        # g_q = (B^T)^+ \cdot g_x
        hes_K = self._energy_hessian - np.tensordot(
            self._internal_gradient, self._cc_to_ic_hessian, axes=1)
        self._internal_hessian = np.dot(
            np.dot(np.linalg.pinv(self._cc_to_ic_gradient.T), hes_K),
            np.linalg.pinv(self._cc_to_ic_gradient))
        # self._tilt_internal_hessian = np.dot(
        #     np.dot(
        #         np.linalg.pinv(self._cc_to_ic_gradient.T),
        #         self._energy_hessian), np.linalg.pinv(self._cc_to_ic_gradient))

    def _regenerate_ic(self):
        self._cc_to_ic_gradient = None
        self._cc_to_ic_hessian = None
        self._clear_g_and_h()
        for ic in self.ic:
            rs = self.coordinates[np.array(ic.atoms)]
            ic.set_new_coordinates(rs)
            d, dd = ic.get_gradient_hessian()
            self._add_cc_to_ic_gradient(d, ic.atoms)  # add gradient
            self._add_cc_to_ic_hessian(dd, ic.atoms)  # add hessian

    def _clear_g_and_h(self):
        self._internal_gradient = None
        self._internal_hessian = None

    def _create_geo_point(self):
        _, x_d, x_dd = self.cost_value_in_cc
        return Point(x_d, x_dd, len(self.numbers))

    def _cost_value(self):
        v, d, dd = self._calculate_cost_value()
        return v, d, dd
        # x_d, x_dd = self._ic_gradient_hessian_transform_to_cc(d, dd)
        # return v, x_d, x_dd

    def _calculate_cost_value(self):
        if self.target_ic is None:
            raise NotSetError("The value of target_ic is not set")
        # initialize function value, gradient and hessian
        value = 0
        deriv = np.zeros(len(self.ic))
        deriv2 = np.zeros((len(self.ic), len(self.ic)), float)
        for i in range(len(self.ic)):
            if self.ic[i].__class__.__name__ in ("BondLength",
                                                 "BendAngle", ):
                v, d, dd = direct_square(self.ic_values[i], self.target_ic[i])
                value += v
                deriv[i] += d
                deriv2[i, i] += dd
        return value, deriv, deriv2

    def _ic_gradient_hessian_transform_to_cc(self, gradient, hessian):
        cartesian_gradient = np.dot(gradient, self._cc_to_ic_gradient)
        cartesian_hessian_part_1 = np.dot(
            np.dot(self._cc_to_ic_gradient.T, hessian),
            self._cc_to_ic_gradient)
        cartesian_hessian_part_2 = np.tensordot(gradient,
                                                self._cc_to_ic_hessian, 1)
        cartesian_hessian = cartesian_hessian_part_1 + cartesian_hessian_part_2
        return cartesian_gradient, cartesian_hessian

    def _check_connectivity(self, atom1, atom2):
        if self.connectivity[atom1, atom2] == 1:
            return True
        elif self.connectivity[atom1, atom2] == 0:
            return False

    def _repeat_check(self, ic_obj):
        for ic in self.ic:
            if ic_obj.atoms == ic.atoms and type(ic_obj) == type(ic):
                return False
        else:
            return True

    def _add_new_internal_coordinate(self, new_ic, d, dd, atoms):
        self._clear_g_and_h()
        self._ic.append(new_ic)
        self._add_cc_to_ic_gradient(d, atoms)  # add gradient
        self._add_cc_to_ic_hessian(dd, atoms)  # add hessian

    def _add_connectivity(self, atoms):
        if len(atoms) != 2:
            raise AtomsNumberError("The number of atoms is not correct")
        num1, num2 = atoms
        self._connectivity[num1, num2] = 1
        self._connectivity[num2, num1] = 1

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
            raise AtomsNumberError("The number of atoms is not correct")
        return tuple(atoms)

    def _add_cc_to_ic_gradient(self, deriv, atoms):  # need to be tested
        if self._cc_to_ic_gradient is None:
            self._cc_to_ic_gradient = np.zeros((0, 3 * len(self.numbers)))
        tmp_vector = np.zeros((1, 3 * len(self.numbers)))
        for i in range(len(atoms)):
            tmp_vector[0, 3 * atoms[i]:3 * atoms[i] + 3] += deriv[i]
        self._cc_to_ic_gradient = np.vstack(
            (self._cc_to_ic_gradient, tmp_vector))

    def _add_cc_to_ic_hessian(self, deriv, atoms):  # need to be tested
        if self._cc_to_ic_hessian is None:
            self._cc_to_ic_hessian = np.zeros(
                (0, 3 * len(self.numbers), 3 * len(self.numbers)))
        tmp_vector = np.zeros(
            (1, 3 * len(self.numbers), 3 * len(self.numbers)))
        for i in range(len(atoms)):
            for j in range(len(atoms)):
                tmp_vector[0, 3 * atoms[i]:3 * atoms[i] + 3, 3 * atoms[j]:3 *
                           atoms[j] + 3] += deriv[i, :3, j]
        self._cc_to_ic_hessian = np.vstack(
            (self._cc_to_ic_hessian, tmp_vector))

# import horton as ht
# fn_xyz = ht.context.get_fn("test/water.xyz")
# mol = ht.IOData.from_file(fn_xyz)  # create a water molecule
# inter = Internal(mol.coordinates, mol.numbers, 0, 1)
# inter.add_bond(0,2)
# inter.add_bond(0,1)
# print (inter._cc_to_ic_hessian.shape)
