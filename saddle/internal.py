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
"internal coordinates implementation"

from copy import deepcopy
from itertools import combinations
from typing import List, Tuple, Union

import numpy as np

from saddle.abclass import CoordinateTypes
from saddle.cartesian import Cartesian
from saddle.coordinate_types import BendCos, BondLength, ConventionDihedral
from saddle.errors import (AtomsIndexError, AtomsNumberError, NotConvergeError,
                           NotSetError)
from saddle.opt import GeoOptimizer, Point
from saddle.periodic.periodic import periodic

__all__ = ('Internal', )


class Internal(Cartesian):
    """Internal Coordinate

    Properties
    ----------
    numbers : np.ndarray(N)
        A numpy array of atomic number for input coordinates
    spin : int
        Spin multiplicity of the molecule
    charge : int
        Charge of the input molecule
    energy : float
        Energy of given Cartesian coordinates system molecule
    energy_gradient : np.ndarray(N)
        Gradient of Energy that calculated through certain method
    energy_hessian : np.ndarray(N, N)
        Hessian of Energy that calculated through cartain method
    coordinates : np.ndarray(N, 3)
        Cartesian information of input molecule
    natom : int
        Number of atoms in the system
    cost_value_in_cc : tuple(float, np.ndarray(K), np.ndarray(K, K))
        Return the cost function value, 1st, and 2nd
        derivative verse cartesian coordinates
    ic : list[CoordinateTypes], len(ic) = K
        A list of CoordinateTypes instance to represent
        internal coordinates information
    ic_values : list[float], len(ic_values) = K
        A list of internal coordinates values
    target_ic : np.ndarray(K,)
        A list of target internal coordinates
    connectivity : np.ndarray(K, K)
        A square matrix represents the connectivity of molecule
        internal coordinates
    b_matrix : np.ndarray(K, 3N)
        Jacobian matrix for transfomr from cartesian coordinates to internal
        coordinates
    internal_gradient : np.ndarray(K,)
        Gradient of energy versus internal coordinates

    Classmethod
    -----------
    from_file(filename, charge=0, spin=1)
        Create cartesian instance from file

    Methods
    -------
    __init__(coordinates, numbers, charge, spin)
        Initializes molecule
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
    delete_ic(*indices)
        delete a exsiting internal coordinates
    set_target_ic(new_ic)
        Set a target internal coordinates to transform into
    set_new_coordinates(new_coor)
        Set molecule with a set of coordinates
    swap_internal_coordinates(index_1, index_2)
        swap the position of two internal coordiantes
    converge_to_target_ic(iteration=100, copy=True)
        Implement optimization process to transform geometry to
        target internal coordinates
    connected_indices(index)
        Return a list of indices that connected to given index
    energy_from_fchk(abs_path, gradient=True, hessian=True):
        Obtain energy and corresponding info from fchk file
    energy_calculation(**kwargs)
        Calculate system energy with different methods through
        software like gaussian
    wipe_ic_info(I_am_sure_i_am_going_to_wipe_all_ic_info)
        wipe internal coordinates information in this structure
    set_new_ics(new_ics)
        Set the internal coordinates depends on the given internal
        coordinates
    print_connectivity()
        print connectivity matrix information on the screen
    auto_select_ic(dihed_special=False)
        automatic internal coordinates depends on buildin algorithm
    """

    def __init__(self, coordinates: 'np.ndarray[float]',
                 numbers: 'np.ndarray[int]', charge: int, spin: int) -> None:
        super(Internal, self).__init__(coordinates, numbers, charge, spin)
        self._ic = []
        # 1 is connected, 0 is not, -1 is itself
        self._connectivity = np.diag([-1] * len(self.numbers))
        self._target_ic = None
        self._cc_to_ic_gradient = None
        self._cc_to_ic_hessian = None
        self._internal_gradient = None
        self._internal_hessian = None
        # self._tilt_internal_hessian = None
        return None

    def add_bond(self, atom1: int, atom2: int) -> None:  # tested
        """Add bond connection between atom1 and atom2

        Arguments
        ---------
        atom1 : int
            the index of the first atom
        atom2 : int
            the index of the second atom
        """
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
        return None

    def add_angle_cos(self, atom1: int, atom2: int,
                      atom3: int) -> None:  # tested
        """Add cos angle connection between atom1, atom2 and atom3. The angle
        is consist of vector(atom1 - atom2) and vector(atom3 - atom2)

        Arguments
        ---------
        atom1 : int
            the index of the first atom
        atom2 : int
            the index of the second(central) atom
        atom3 : int
            the index of the third atom
        """
        if atom1 == atom3:
            raise AtomsIndexError("The two indece are the same")
        atoms = (atom1, atom2, atom3)
        atoms = self._atoms_sequence_reorder(atoms)
        rs = self.coordinates[np.array(atoms)]
        new_ic_obj = BendCos(atoms, rs)
        d, dd = new_ic_obj.get_gradient_hessian()
        # check if the angle is formed by two connected bonds
        if self._check_connectivity(atom1, atom2) and self._check_connectivity(
                atom2, atom3):
            if self._repeat_check(new_ic_obj):
                self._add_new_internal_coordinate(new_ic_obj, d, dd, atoms)
        return None

    def add_dihedral(self, atom1: int, atom2: int, atom3: int,
                     atom4: int) -> None:  # tested
        """Add a dihedral connection for atom1, atom2, atom3, and atom4
        The dihedral is consist of plane(atom1, atom2, atom3) and
        plane(atom2, atom3, atom4)

        Arguments
        ---------
        atom1 : int
            index of the first atom
        atom2 : int
            index ot the second atom
        atom3 : int
            index of the third atom
        atom4 : int
            index of the fourth atom
        """
        if atom1 == atom4 or atom2 == atom3:
            raise AtomsIndexError("The two indece are the same")
        atoms = (atom1, atom2, atom3, atom4)
        atoms = self._atoms_sequence_reorder(atoms)
        rs = self.coordinates[np.array(atoms)]
        new_ic_obj = ConventionDihedral(atoms, rs)
        d, dd = new_ic_obj.get_gradient_hessian()
        if (self._check_connectivity(atom2, atom3)
                and (self._check_connectivity(atom1, atom2)
                     or self._check_connectivity(atom1, atom3))
                and (self._check_connectivity(atom4, atom3)
                     or self._check_connectivity(atom4, atom2))):
            if self._repeat_check(new_ic_obj):
                self._add_new_internal_coordinate(new_ic_obj, d, dd, atoms)
        return None

    def delete_ic(self, *indices: int) -> None:
        """Delete a exsiting internal coordinates

        Arguments
        ---------
        *indices : int
            The index of each internal coordinate
        """
        indices = np.sort(np.array(indices))
        assert len(indices) <= len(self.ic)
        assert np.max(indices) < len(self.ic)
        for seq, index in enumerate(indices):
            self._delete_ic_index(index - seq)
        return None

    def set_target_ic(self, new_ic: 'np.ndarray[float]') -> None:
        """Set a target internal coordinates to optimize

        Arguments
        ---------
        new_ic : np.ndarray(K,) or list of int, len(new_ic) = K
        """
        if len(new_ic) != len(self.ic):
            raise AtomsNumberError("The ic is not in the same shape")
        self._target_ic = np.array(new_ic).copy()
        return None

    def set_new_coordinates(self,
                            new_coor: 'np.ndarray') -> None:  # to be tested
        """Assign new cartesian coordinates to this molecule

        Arguments
        ---------
        new_coor : np.ndarray(N, 3)
            New cartesian coordinates of the system
        """
        super(Internal, self).set_new_coordinates(new_coor)
        self._regenerate_ic()
        return None

    def swap_internal_coordinates(self, index_1: int, index_2: int) -> None:
        """Swap two internal coordinates sequence

        Arguments
        ---------
        index_1 : int
            index of the first internal coordinate
        index_2 : int
            index of the second internal coordinate
        """
        self._ic[index_1], self._ic[index_2] = self._ic[index_2], self._ic[
            index_1]
        self._regenerate_ic()
        return None

    def converge_to_target_ic(self,
                              iteration: int = 100) -> None:  # to be test
        """Using buildin optimization process to optimize geometry to target
        internal coordinates as close as possible

        Arguments
        ---------
        iteration : int, iteration > 0, default is 100
            number of iteration for optimization process
        """
        optimizer = GeoOptimizer()
        init_point = self._create_geo_point()
        optimizer.add_new(init_point)
        for _ in range(iteration):
            optimizer.tweak_hessian(optimizer.newest)
            step = optimizer.trust_radius_step(optimizer.newest)
            new_coor = self.coordinates + step.reshape(-1, 3)
            self.set_new_coordinates(new_coor)
            new_point = self._create_geo_point()
            optimizer.add_new(new_point)
            if optimizer.converge(optimizer.newest):
                print("finished")
                return None
            optimizer.update_trust_radius(optimizer.newest)
        raise NotConvergeError("The optimization failed to converge")

    def connected_indices(self, index: int) -> 'np.ndarray[int]':
        """Return the indices of atoms connected to given index atom

        Arguments
        ---------
        index : int
            the index of given index for finding connection

        Returns
        -------
        connected_index : np.ndarray(M)
            indices of atoms connected to given index
        """
        connection = self.connectivity[index]
        connected_index = np.where(connection > 0)[0]
        return connected_index

    def energy_from_fchk(self,
                         abs_path: str,
                         gradient: bool = True,
                         hessian: bool = True):
        """Abtain Energy and relative information from FCHK file.

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
        super(Internal, self).energy_from_fchk(abs_path, gradient, hessian)
        self._energy_hessian_transformation()

    def energy_calculation(self, *_, title: str, method: str) -> None:
        """Conduct calculation with designated method.

        Keywords Arguments
        ------------------
        title : str, default is 'untitled'
            title of input and out put name without postfix
        method : str, default is 'g09'
            name of the program(method) used to calculate energy and other
            property
        """
        super(Internal, self).energy_calculation(title=title, method=method)
        self._energy_hessian_transformation()
        return None
        # h_q = (B^T)^+ \cdot (H_x - K) \cdot B^+

    def wipe_ic_info(self,
                     I_am_sure_i_am_going_to_wipe_all_ic_info: bool) -> None:
        """wipe all internal coordinates information in this structure
        including ic, ic_values, target_ic, b_matrix, internal gradient and
        internal hessian

        Arguments
        ---------
        I_am_sure_i_am_going_to_wipe_all_ic_info : bool
            Double check for wipe important ic info. True for confirm,
            otherwise False
        """
        if I_am_sure_i_am_going_to_wipe_all_ic_info:
            self._clear_ic_info()
        return None

    def set_new_ics(self, new_ics: List[CoordinateTypes]) -> None:
        """Set the internal coordinates to the given one

        Arguments
        ---------
        new_ics : list
            The list of coordinates as the template
        """
        assert all(isinstance(ic, CoordinateTypes) for ic in new_ics)
        self.wipe_ic_info(True)
        self._ic = deepcopy(list(new_ics))
        self._regenerate_ic()
        self._regenerate_connectivity()
        return None

    @property
    def cost_value_in_cc(
            self) -> Tuple[float, 'np.ndarray[float]', 'np.ndarray[float]']:
        """Cost function value and its gradient, hessian versus Cartesian
        coordinates

        Returns
        v, x_d, x_dd : tuple(float, np.ndarray(3N,), np.ndarray(3N, 3N))
            v, the value of cost function
            x_d, the gradient vs cartesian coordinates
            x_dd, the hessian vs cartesian coordinates
        """
        v, d, dd = self._cost_value()
        x_d, x_dd = self._ic_gradient_hessian_transform_to_cc(d, dd)
        return v, x_d, x_dd

    @property
    def ic(self) -> List[CoordinateTypes]:
        """list of internal coordinates object

        Returns
        -------
        ic : list of coordinate_types, len(ic) = K
        """
        return self._ic

    @property
    def ic_values(self) -> 'np.ndarray[float]':
        """list of internal coordinates values

        Returns
        -------
        ic_values : list of float, len(ic_values) = K
        """
        value = [i.value for i in self._ic]
        return np.array(value)

    @property
    def target_ic(self) -> 'np.ndarray[float]':
        """target internal coordinates values

        Returns
        -------
        target_ic : np.ndarray(K,)
        """
        return self._target_ic

    @property
    def connectivity(self) -> 'np.ndarray[float]':
        """A connectivity matrix shows the connection of atoms, 1 is
        connected, 0 is not connected, -1 is itself

        Returns
        -------
        connectivity : np.ndarray(K, K)
        """
        return self._connectivity

    @property
    def b_matrix(self) -> 'np.ndarray[float]':
        """Jacobian matrix for transforming cartisian coordinates into
        internal coordinates

        Returns
        -------
        b_matrix : np.ndarray(K, 3N)
        """
        return self._cc_to_ic_gradient

    @property
    def internal_gradient(self) -> 'np.ndarray[float]':
        """Gradient of energy versus internal coordinates

        Returns
        -------
        internal_gradient : np.ndarray(K,)
        """
        return self._internal_gradient

    def print_connectivity(self) -> None:
        """Print the connectivity matrix on screen
        """
        format_func = "{:3}".format
        print("--Connectivity Starts-- \n")
        for i, _ in enumerate(self.numbers):
            print(" ".join(map(format_func, self.connectivity[i, :i + 1])))
            print("\n--Connectivity Ends--")
        return None

    def auto_select_ic(self,
                       dihed_special: bool = False,
                       reset_ic: bool = True,
                       keep_bond: bool = False) -> None:
        """A method for Automatically selecting internal coordinates based on
        out buildin algorithm

        Arguments
        ---------
        dihed_special : bool, default is False
            choice of special dihedral indicator for dealing with collinear
            problem. True for enable, otherwise False
        reset_ic : bool, default is True
            wipe all the existing internal coordinates, regenerate all the
            internal coordinates. True for enable, otherwise False
        keep_bond : bool, default is False
            keep bond information and regenerate bend angle and dihedral
            information.
        """
        if dihed_special:
            raise NotImplementedError(
                "This functionality hasn't been implemented yet")
        bonds = [i for i in self.ic if isinstance(i, BondLength)]
        if reset_ic is True:
            self.wipe_ic_info(True)
        if keep_bond is False:
            self._auto_select_bond()
        else:
            self.set_new_ics(bonds)
        self._auto_select_angle()
        self._auto_select_dihed_normal()
        self._auto_select_dihed_improper()
        self._recal_g_and_h()
        return None

    def _delete_ic_index(self, index: int) -> None:
        del self._ic[index]
        self._regenerate_ic()
        self._regenerate_connectivity()
        return None

    def _clear_ic_info(self) -> None:  # tested
        """Wipe all the internal information in this structure
        """
        self._ic = []
        self._connectivity = np.diag([-1] * len(self.numbers))
        self._target_ic = None
        self._cc_to_ic_gradient = None
        self._cc_to_ic_hessian = None
        self._internal_gradient = None
        self._internal_hessian = None
        return None

    def _auto_select_bond(self) -> None:
        """A private method for automatically selecting bond
        """
        halidish_atom = set([7, 8, 9, 15, 16, 17])
        all_halo_index = (i for i, j in enumerate(self.numbers)
                          if j in halidish_atom)
        for index_i, index_j in combinations(range(len(self.numbers)), 2):
            atom_num1 = self.numbers[index_i]
            atom_num2 = self.numbers[index_j]
            distance = self.distance(index_i, index_j)
            radius_sum = (periodic[atom_num1].cov_radius +
                          periodic[atom_num2].cov_radius)
            if distance < 1.3 * radius_sum:
                self.add_bond(index_i, index_j)
                # test hydrogen bond
                if atom_num1 == 1 and atom_num2 in halidish_atom:
                    h_index = index_i
                    halo_index = index_j
                elif atom_num2 == 1 and atom_num1 in halidish_atom:
                    h_index = index_j
                    halo_index = index_i
                else:
                    continue
                potent_halo_index = (i for i in all_halo_index
                                     if i != halo_index)  # all other halo
                for index_k in potent_halo_index:
                    dis = self.distance(h_index, index_k)
                    angle = self.angle(halo_index, h_index, index_k)
                    thresh_sum = (periodic[self.numbers[h_index]].vdw_radius +
                                  periodic[self.numbers[index_k]].vdw_radius)
                    if dis <= 0.9 * thresh_sum and angle >= 1.5708:
                        self.add_bond(h_index, index_k)  # add H bond
        return None

    def _auto_select_angle(self) -> None:
        """A private method for automatically selecting angle
        """
        for center_index, _ in enumerate(self.numbers):
            connected = self.connected_indices(center_index)
            if len(connected) >= 2:
                for side_1, side_2 in combinations(connected, 2):
                    self.add_angle_cos(side_1, center_index, side_2)
        return None

    def _auto_select_dihed_normal(self) -> None:
        """A private method for automatically selecting normal dihedral
        """
        for center_ind_1, _ in enumerate(self.numbers):
            # find indices connected to center_ind_1
            connected = self.connected_indices(center_ind_1)
            if len(connected) >= 2:
                for center_ind_2 in connected:
                    sum_cnct = np.sum(self.connectivity, axis=0)
                    # find total connection for all atoms connected c1
                    sum_select_cnct = sum_cnct[connected]
                    sorted_index = sum_select_cnct.argsort()[::-1]
                    side_1 = connected[sorted_index[0]]
                    # select the atom with the largest connection
                    if connected[sorted_index[0]] == center_ind_2:
                        side_1 = connected[sorted_index[1]]
                    connected_to_index_2 = self.connected_indices(center_ind_2)
                    for side_2 in connected_to_index_2:
                        if side_2 not in (center_ind_1, center_ind_2, side_1):
                            self.add_dihedral(side_1, center_ind_1,
                                              center_ind_2, side_2)
        return None

    def _auto_select_dihed_improper(self) -> None:
        """A private method for automatically selecting improper dihedral
        """
        connect_sum = np.sum(
            self.connectivity, axis=0) + 1  # cancel -1 for itself
        for center_ind, _ in enumerate(connect_sum):
            if connect_sum[center_ind] >= 3:
                cnct_atoms = self.connected_indices(center_ind)
                cnct_total = len(cnct_atoms)
                for i, j, k in combinations(range(cnct_total), 3):
                    ind_i, ind_j, ind_k = cnct_atoms[[i, j, k]]
                    ang1_r = self.angle(ind_i, center_ind, ind_j)
                    ang2_r = self.angle(ind_i, center_ind, ind_k)
                    ang3_r = self.angle(ind_j, center_ind, ind_k)
                    sum_r = ang1_r + ang2_r + ang3_r
                    if sum_r >= 6.02139:
                        self.add_dihedral(ind_i, center_ind, ind_j, ind_k)
        return None

    def _energy_hessian_transformation(self) -> None:
        """convert gradient, hessian versus cartesian coordinates into
        gradient, hessian versus internal coordinates
        ..math::
            g_q = (B_T)^+ g_x
            H_q = B_T^+ (H_x - K) B^+ + K, where
            K = g_q b^\\prime
        """
        self._internal_gradient = np.dot(
            np.linalg.pinv(self._cc_to_ic_gradient.T), self._energy_gradient)
        # g_q = (B^T)^+ \cdot g_x
        hes_K = self._energy_hessian - np.tensordot(
            self._internal_gradient, self._cc_to_ic_hessian, axes=1)
        self._internal_hessian = np.dot(
            np.dot(np.linalg.pinv(self._cc_to_ic_gradient.T), hes_K),
            np.linalg.pinv(self._cc_to_ic_gradient))
        # self._tilt_internal_hessian = np.dot(
        #   np.dot(
        #     np.linalg.pinv(self._cc_to_ic_gradient.T),
        #     self._energy_hessian), np.linalg.pinv(self._cc_to_ic_gradient))
        return None

    def _regenerate_ic(self) -> None:
        """reset internal coordinates system, reset gradient, hessian versus
        internal coordinates, regenerate internal coordinates and
        transformation matrix
        """
        self._cc_to_ic_gradient = None
        self._cc_to_ic_hessian = None
        for ic in self.ic:
            rs = self.coordinates[np.array(ic.atoms)]
            ic.set_new_coordinates(rs)
            d, dd = ic.get_gradient_hessian()
            self._add_cc_to_ic_gradient(d, ic.atoms)  # add transform gradient
            self._add_cc_to_ic_hessian(dd, ic.atoms)  # add transform hessian
        self._recal_g_and_h()  # clean internal gradient and hessian
        return None

    def _regenerate_connectivity(self) -> None:
        """regenerate the connectivity of molecule depends on present
        internal coordinates
        """
        self._connectivity = np.diag([-1] * len(self.numbers))
        for ic in self.ic:
            if isinstance(ic, BondLength):
                self._add_connectivity(ic.atoms)
        return None

    def _recal_g_and_h(self) -> None:
        """reset internal energy gradient and hessian matrix
        """
        self._internal_gradient = None
        self._internal_hessian = None
        if (self._energy_gradient is not None
                and self._energy_hessian is not None):
            self._energy_hessian_transformation()
        return None

    def _create_geo_point(self) -> Point:
        """create a Point object based on self internal coordinates to undergo
        a optimizatino process in order to converge to target_ic

        Returns
        -------
        geo_point : Point object
        """
        _, x_d, x_dd = self.cost_value_in_cc
        return Point(x_d, x_dd, len(self.numbers))

    def _cost_value(
            self) -> Tuple[float, 'np.ndarray[float]', 'np.ndarray[float]']:
        """Calculate value of cost function as well as its gradient and
        hessian versus internal coordinates

        Returns
        -------
        value, deriv, deriv2 : tuple(float, np.ndarray(K,), np.ndarray(K, K))
            value, the value of cost function
            deriv, the gradient vs internal coordinates
            deriv2, the hessian vs internal coordinates
        """
        if self.target_ic is None:
            raise NotSetError("The value of target_ic is not set")
        # initialize function value, gradient and hessian
        value = 0
        deriv = np.zeros(len(self.ic))
        deriv2 = np.zeros((len(self.ic), len(self.ic)), float)
        for i, _ in enumerate(self.ic):
            if self.ic[i].__class__.__name__ in ("BondLength", "BendCos",
                                                 "BendAngle"):
                v, d, dd = self._direct_square(self.ic_values[i],
                                               self.target_ic[i])
                value += v
                deriv[i] += d
                deriv2[i, i] += dd
        return value, deriv, deriv2

    def _ic_gradient_hessian_transform_to_cc(
            self, gradient: 'np.ndarray[float]', hessian: 'np.ndarray[float]'
    ) -> Tuple['np.ndarray[float]', 'np.ndarray[float]']:
        """transform energy gradient and hessian back from internal coordinates to
        cartesian cooridnates
        ..math::
            g_x = B_T g_q
            H_x = B_T H_q B + K, where
            K = g_q b^\\prime

        Returns
        -------
        cartesian_gradient, cartesian_hessian : tuple(np.ndarray(3N,),
                                                      np.ndarray(3N, 3N))
            cartesian_gradient, energy gradient vs cartesian coordinates
            cartesian_hessian, energy hessian vs cartesian coordinates
        """
        cartesian_gradient = np.dot(gradient, self._cc_to_ic_gradient)
        cartesian_hessian_part_1 = np.dot(
            np.dot(self._cc_to_ic_gradient.T, hessian),
            self._cc_to_ic_gradient)
        cartesian_hessian_part_2 = np.tensordot(gradient,
                                                self._cc_to_ic_hessian, 1)
        cartesian_hessian = cartesian_hessian_part_1 + cartesian_hessian_part_2
        return cartesian_gradient, cartesian_hessian

    def _check_connectivity(self, atom1: int, atom2: int) -> bool:
        """Check whether two atoms are connected or not

        Arguments
        ---------
        atom1 : int
            The index of atom1
        atom2 : int
            The index of atom2

        Returns
        -------
        connected : bool
            Return True if they are connected, otherwise False
        """
        if self.connectivity[atom1, atom2] == 1:
            return True
        elif self.connectivity[atom1, atom2] == 0:
            return False

    def _repeat_check(self, ic_obj: CoordinateTypes) -> bool:
        """Check whether the given ic_obj already existed in ic list or not

        Arguments
        ---------
        ic_obj : Coordinate_Types
            the given ic object to be tested

        Returns
        -------
        repeat_check : bool
            Return True if there is no duplicate and it's a valid new ic
            object, otherwise False
        """
        for ic in self.ic:
            if ic_obj.atoms == ic.atoms and type(ic_obj) == type(ic):
                return False
        return True

    def _add_new_internal_coordinate(
            self, new_ic: CoordinateTypes, d: 'np.ndarray[float]',
            dd: "np.ndarray[float]", atoms: Tuple[int, ...]) -> None:
        """Add a new ic object to the system and add corresponding
        transformation matrix parts
        """
        self._ic.append(new_ic)
        self._add_cc_to_ic_gradient(d, atoms)  # add gradient
        self._add_cc_to_ic_hessian(dd, atoms)  # add hessian
        self._recal_g_and_h()
        return None

    def _add_connectivity(self, atoms: Tuple[int, ...]) -> None:
        """Change the value of connectivity matrix to 1 for two atoms

        Arguments
        ---------
        atoms : list or tuple of int, len(atoms) = 2
        """
        if len(atoms) != 2:
            raise AtomsNumberError("The number of atoms is not correct")
        num1, num2 = atoms
        self._connectivity[num1, num2] = 1
        self._connectivity[num2, num1] = 1

    @staticmethod
    def _atoms_sequence_reorder(atoms: Tuple[int, ...]) -> None:
        """Change the atoms in each ic object in ascending sequence without
        changing its representative

        Arguments
        ---------
        atoms : list or tuple of int, 2 <= len(atoms) <= 4
            indices of atoms for certain ic object

        Returns
        -------
        atoms : tuple of int
            reordered sequence of indices of atoms for same ic object
        """
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

    def _add_cc_to_ic_gradient(
            self, deriv: 'np.ndarray[float]',
            atoms: Tuple[int, ...]) -> None:  # need to be tested
        """Add new entries from a new ic to transformation matrix gradient

        Arguments
        ---------
        deriv : np.ndarray(3N,)
            tranformation gradient matrix given ic regarding to given atoms
        atoms : list or tuple of int
            indices of atoms for those transformation
        """
        if self._cc_to_ic_gradient is None:
            self._cc_to_ic_gradient = np.zeros((0, 3 * len(self.numbers)))
        tmp_vector = np.zeros((1, 3 * len(self.numbers)))
        for i, _ in enumerate(atoms):
            tmp_vector[0, 3 * atoms[i]:3 * atoms[i] + 3] += deriv[i]
        self._cc_to_ic_gradient = np.vstack((self._cc_to_ic_gradient,
                                             tmp_vector))
        return None

    def _add_cc_to_ic_hessian(
            self, deriv: 'np.ndarray[float]',
            atoms: Tuple[int, ...]) -> None:  # need to be tested
        """Add new entries from a new ic to transformation matrix hessian

        Arguments
        ---------
        deriv : np.ndarray(3N, 3N)
            tranformation hessian matrix given ic regarding to given atoms
        atoms : list or tuple of int
            indices of atoms for those transformation
        """
        if self._cc_to_ic_hessian is None:
            self._cc_to_ic_hessian = np.zeros((0, 3 * len(self.numbers),
                                               3 * len(self.numbers)))
        tmp_vector = np.zeros((1, 3 * len(self.numbers),
                               3 * len(self.numbers)))
        for i, _ in enumerate(atoms):
            for j, _ in enumerate(atoms):
                tmp_vector[0, 3 * atoms[i]:3 * atoms[i] + 3, 3 * atoms[j]:
                           3 * atoms[j] + 3] += deriv[i, :3, j]
        self._cc_to_ic_hessian = np.vstack((self._cc_to_ic_hessian,
                                            tmp_vector))
        return None

    @staticmethod
    def _direct_square(origin: float, target: float) -> Tuple[float, ...]:
        """Calculate cost function and it's derivatives for geometry transiformation

        Arguments
        ---------
        origin : float
            The value of original internal coordinate
        target : float
            The value of the target internal coordinate

        Returns
        -------
        (value, deriv, deriv2) : ()
        """
        value = (origin - target)**2
        deriv = 2 * (origin - target)
        deriv2 = 2
        return value, deriv, deriv2
