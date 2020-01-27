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
"""internal coordinates implementation."""
# import warnings
from copy import deepcopy
from heapq import heappop, heappush
from itertools import combinations
from typing import List, Tuple

from saddle.cartesian import Cartesian
from saddle.coordinate_types import (
    BendAngle,
    BondLength,
    CoordinateTypes,
    DihedralAngle,
    NewDihedralCross,
    NewDihedralDot,
)
from saddle.errors import (
    AtomsIndexError,
    AtomsNumberError,
    NotConvergeError,
    NotSetError,
)
from saddle.math_lib import pse_inv
from saddle.molmod import bend_angle, dihed_angle
from saddle.opt import GeoOptimizer, Point
from saddle.periodic.periodic import periodic

import numpy as np
from numpy import cos, sin

# from saddle.utils import deprecated

__all__ = ("Internal",)


class Internal(Cartesian):
    """Internal Coordinate Class.

    Properties
    ----------
    numbers : np.ndarray(N)
        A numpy array of atomic number for input coordinates
    multi : int
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
        derivative versus cartesian coordinates
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
        Jacobian matrix for transforming from cartesian coordinates to internal
        coordinates
    internal_gradient : np.ndarray(K,)
        Gradient of energy versus internal coordinates

    Classmethod
    -----------
    from_file(filename, charge=0, multi=1)
        Create cartesian instance from file

    Methods
    -------
    __init__(coordinates, numbers, charge, multi)
        Initializes molecule
    distance(index1, index2)
        Calculate distance between two atoms with index1 and index2
    angle(index1, index2, index3)
        Calculate angle between atoms with index1, index2, and index3
    add_bond(atom1, atom2)
        Add a bond between atom1 and atom2
    add_angle(atom1, atom2, atom3)
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

    def __init__(
        self,
        coordinates: "np.ndarray[float]",
        numbers: "np.ndarray[int]",
        charge: int,
        multi: int,
        title: str = "",
    ) -> None:
        super(Internal, self).__init__(coordinates, numbers, charge, multi, title)
        self._ic = []
        # 1 is connected, 0 is not, -1 is itself
        self._connectivity = np.diag([-1] * len(self.numbers))
        self._cc_to_ic_gradient = None
        self._cc_to_ic_hessian = None

        # to indicate fragment groups in system
        self._fragment = np.arange(self.natom)

    def add_bond(
        self, atom1: int, atom2: int, *_, b_type: int = 1, weight: int = 1
    ) -> None:  # tested
        """Add bond connection between atom1 and atom2.

        Arguments
        ---------
        atom1 : int
            the index of the first atom
        atom2 : int
            the index of the second atom
        """
        if atom1 == atom2:
            raise AtomsIndexError("The two indices are the same")
        atoms = (atom1, atom2)
        # reorder the sequence of atoms indices
        atoms = self._atoms_sequence_reorder(atoms)
        # just sorting, no sequence changes
        if self._repeat_atoms_check(atoms):
            rs = self.coordinates[np.array(atoms)]
            new_ic_obj = BondLength(atoms, rs, ic_type=b_type, weight=weight)
            d, dd = new_ic_obj.get_gradient_hessian()
            # gradient and hessian need to be set
            self._add_new_internal_coordinate(new_ic_obj, d, dd, atoms)
            # after adding a bond, change the connectivity of atoms pair to 1
            self._add_connectivity(atoms, b_type)
            if b_type == 1:
                self._allocate_fragment_group(atom1, atom2)

    def add_angle(
        self, atom1: int, atom2: int, atom3: int, weight: int = 1
    ) -> None:  # tested
        """Add cos angle connection between atom1, atom2 and atom3.

        The angle is consist of vector(atom1 - atom2) and vector(atom3 - atom2)

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
        if self._repeat_atoms_check(atoms):
            if self._check_connectivity(atom1, atom2) and self._check_connectivity(
                atom2, atom3
            ):  # check if the angle is formed by two connected bonds
                rs = self.coordinates[np.array(atoms)]
                new_ic_obj = BendAngle(atoms, rs, weight=weight)
                d, dd = new_ic_obj.get_gradient_hessian()
                self._add_new_internal_coordinate(new_ic_obj, d, dd, atoms)

    def add_dihedral(
        self,
        atom1: int,
        atom2: int,
        atom3: int,
        atom4: int,
        weight: int = 1,
        *_,
        special=False,
    ) -> None:  # tested
        """Add a dihedral connection for atom1, atom2, atom3, and atom4.

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
        # atoms = self._atoms_sequence_reorder(atoms)
        if self._repeat_atoms_check(atoms):
            if (
                self._check_connectivity(atom2, atom3)
                and (
                    self._check_connectivity(atom1, atom2)
                    or self._check_connectivity(atom1, atom3)
                )
                and (
                    self._check_connectivity(atom4, atom3)
                    or self._check_connectivity(atom4, atom2)
                )
            ):
                rs = self.coordinates[np.array(atoms)]
                if special:
                    # add Dot dihedral
                    new_ic_obj_1 = NewDihedralDot(atoms, rs, weight=weight)
                    d, dd = new_ic_obj_1.get_gradient_hessian()
                    self._add_new_internal_coordinate(new_ic_obj_1, d, dd, atoms)
                    # add cross dihedral
                    new_ic_obj_2 = NewDihedralCross(atoms, rs, weight=weight)
                    d, dd = new_ic_obj_2.get_gradient_hessian()
                    self._add_new_internal_coordinate(new_ic_obj_2, d, dd, atoms)
                else:
                    new_ic_obj = DihedralAngle(atoms, rs, weight=weight)
                    d, dd = new_ic_obj.get_gradient_hessian()
                    self._add_new_internal_coordinate(new_ic_obj, d, dd, atoms)

    def delete_ic(self, *indices: int) -> None:
        """Delete an exsiting internal coordinates.

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

    def set_target_ic(self, new_ic: "np.ndarray[float]") -> None:
        """Set a target internal coordinates to optimize.

        Arguments
        ---------
        new_ic : np.ndarray(K,) or list of int, len(new_ic) = K
        """
        if len(new_ic) != len(self.ic):
            raise AtomsNumberError("The ic is not in the same shape")
        for index, ic in enumerate(self.ic):
            ic.target = new_ic[index]

    def set_new_coordinates(self, new_coor: "np.ndarray") -> None:
        """Assign new cartesian coordinates to this molecule.

        Arguments
        ---------
        new_coor : np.ndarray(N, 3)
            New cartesian coordinates of the system483G
        """
        super(Internal, self).set_new_coordinates(new_coor)
        self._regenerate_ic()

    def swap_internal_coordinates(self, index_1: int, index_2: int) -> None:
        """Swap two internal coordinates sequence.

        Arguments
        ---------
        index_1 : int
            index of the first internal coordinate
        index_2 : int
            index of the second internal coordinate
        """
        self._ic[index_1], self._ic[index_2] = self._ic[index_2], self._ic[index_1]
        self._regenerate_ic()

    # @deprecated("Use 'optimize_to_target_ic' instead")
    def converge_to_target_ic(
        self, max_iter: int = 100, ignore_dihed=False, flex_sin=True
    ) -> None:  # to be test
        """Using buildin optimization process to optimize geometry to target ic.

        Arguments
        ---------
        max_iter : int, max_iter > 0, default is 100
            number of iteration for optimization process
        """
        optimizer = GeoOptimizer()
        wts_bk = None
        # calculate the init structure
        init_coor = (
            np.dot(pse_inv(self.b_matrix), self.target_ic - self.ic_values).reshape(
                -1, 3
            )
            + self.coordinates
        )
        # ignore dihedral
        if ignore_dihed:
            wts_bk = self.ic_weights
            self.set_dihed_weights(0)
        self.set_new_coordinates(init_coor)
        init_point = self.create_geo_point(flex_sin=flex_sin)
        optimizer.add_new(init_point)
        for _ in range(max_iter):
            optimizer.tweak_hessian(optimizer.newest)
            step = optimizer.trust_radius_step(optimizer.newest)
            new_coor = self.coordinates + step.reshape(-1, 3)
            self.set_new_coordinates(new_coor)
            new_point = self.create_geo_point(flex_sin=flex_sin)
            optimizer.add_new(new_point)
            if optimizer.converge(optimizer.newest):
                # print("finished")
                return
            optimizer.update_trust_radius(optimizer.newest)
        if wts_bk is not None:
            self.set_ic_weights(wts_bk)
        raise NotConvergeError(
            "The coordinates transformation optimization failed to converge"
        )

    def connected_indices(self, index: int, *_, exclude=None) -> "np.ndarray[int]":
        """Return the indices of atoms connected to given index atom.

        Arguments
        ---------
        index : int
            the index of given index for finding connection
        exclude : int
            the value of connectivity be excluded from selection

        Returns
        -------
        connected_index : np.ndarray[int]
            indices of atoms connected to given index
        """
        if exclude:
            assert isinstance(exclude, int)
            assert exclude > 0
        connection = self.connectivity[index]
        connected_index = np.where((connection > 0) & (connection != exclude))[0]
        return connected_index

    @property
    def number_of_connect(self):
        """Return a list of number of connections of each atoms in the molecule.

           The index of the list is the same index in self.numbers. Connection
           exclude auxiliary bond
        """
        return np.sum((self.connectivity > 0) & (self.connectivity != 4), axis=0)

    def energy_from_fchk(
        self, abs_path: str, *_, gradient: bool = True, hessian: bool = True
    ):
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

    def energy_calculation(self, *_, method: str = "g09") -> None:
        """Conduct calculation with designated method.

        Keywords Arguments
        ------------------
        title : str, default is 'untitled'
            title of input and out put name without postfix
        method : str, default is 'g09'
            name of the program(method) used to calculate energy and other
            property
        """
        super(Internal, self).energy_calculation(method=method)
        # h_q = (B^T)^+ \cdot (H_x - K) \cdot B^+

    def wipe_ic_info(self, I_am_sure_i_am_going_to_wipe_all_ic_info: bool) -> None:
        """wipe all internal coordinates information in this structure.

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

    def set_new_ics(self, new_ics: List[CoordinateTypes]) -> None:
        """Set the internal coordinates to the given one.

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

    def set_ic_weights(self, weights):
        """Set a new set of internal coordinates weights.

        Parameters
        ----------
        weights : np.ndarray(N,)
            weights array of the same size as internal coordinates

        Raises
        ------
        TypeError
            weights need to be of type np.ndarray
        ValueError
            the number of elements in weights need to be the same as IC
        """
        if not isinstance(weights, np.ndarray):
            raise TypeError(f"Weights need to be a numpy array, got {type(weights)}")
        if len(weights) != len(self.ic):
            raise ValueError(f"# of Weights is not equal to ics. Total ")
        for i, ic in enumerate(self.ic):
            ic.weight = weights[i]

    def set_dihed_weights(self, value):
        """Give a new set of weight for dihedral coordinates.

        Parameters
        ----------
        value : float
            new weight for dihedral, value need to be >= 0
        """
        for ic in self.ic:
            if isinstance(ic, DihedralAngle):
                ic.weight = value

    @property
    def ic_weights(self):
        """np.ndarray(N,): get internal coordinates' weights."""
        return np.array([i.weight for i in self.ic])

    @property
    def cost_value_in_cc(
        self,
    ) -> Tuple[float, "np.ndarray[float]", "np.ndarray[float]"]:
        """Cost function value and its gradient, hessian in Cartesian coordinates.

        Returns
        v, x_d, x_dd : tuple(float, np.ndarray(3N,), np.ndarray(3N, 3N))
            v, the value of cost function
            x_d, the gradient vs cartesian coordinates
            x_dd, the hessian vs cartesian coordinates
        """
        q_d, q_dd = self._cost_q_d, self._cost_q_dd
        x_d, x_dd = self._ic_gradient_hessian_transform_to_cc(q_d, q_dd)
        return self.tf_cost, x_d, x_dd

    @property
    def tf_cost(self):
        """Compute cost function for transformation."""
        return self._compute_tfm_cost()

    @property
    def _cost_q_d(self):
        """Compute tf cost function gradient."""
        return np.array([i.cost_d for i in self.ic])

    @property
    def _cost_q_dd(self):
        """Compute tf cost function hessian."""
        return np.diag([i.cost_dd for i in self.ic])

    def _compute_tfm_cost(self):
        """Compute tf cost function."""
        return sum([i.cost_v for i in self.ic])

    def _compute_tfm_gradient_internal(self):
        """Compute tf cost function gradient in internal coors (Obselete)."""
        cost_g_q = np.zeros(len(self.ic))
        for index, ic in enumerate(self.ic):
            if isinstance(ic, (BondLength, NewDihedralCross, NewDihedralDot)):
                cost_g_q[index] += 2 * (ic.value - ic.target) * ic.weight
            elif isinstance(ic, BendAngle):
                cost_g_q[index] += (
                    -2 * (cos(ic.value) - cos(ic.target)) * sin(ic.value) * ic.weight
                )
            elif isinstance(ic, DihedralAngle):
                continue
            else:
                raise TypeError(f"Given {ic} is not support type")
        return cost_g_q

    def _compute_tfm_gradient(self):
        """function used to compute transform gradient(obselete).

        Returns
        -------
        TYPE
            Description
        """
        cost_g_x = np.zeros(self.natom * 3)
        for index, ic in enumerate(self.ic):
            if isinstance(ic, DihedralAngle):
                atoms = np.array(ic.atoms)
                # value and b matrix of angle1
                ang1_ind = np.array(atoms[:3])
                ang1_v, ang1_d1 = bend_angle(self.coordinates[ang1_ind], 1)
                # value and b matrix of angle2
                ang2_ind = np.array(atoms[1:])
                ang2_v, ang2_d1 = bend_angle(self.coordinates[ang2_ind], 1)
                # value and b matrix for dihed
                _, dihed_d1 = dihed_angle(self.coordinates[atoms], 1)
                # derivative of angle 1
                tmp_cost_g_q = (
                    2
                    * sin(ang1_v)
                    * cos(ang1_v)
                    * sin(ang2_v) ** 2
                    * (2 - 2 * cos(ic.value - ic.target))
                    * ic.weight
                )
                tmp_cost_g_x = ang1_d1 * tmp_cost_g_q
                cost_g_x.reshape(-1, 3)[ang1_ind] += tmp_cost_g_x

                # derivative of angle 2
                tmp_cost_g_q = (
                    2
                    * sin(ang2_v)
                    * cos(ang2_v)
                    * sin(ang1_v) ** 2
                    * (2 - 2 * cos(ic.value - ic.target))
                    * ic.weight
                )
                tmp_cost_g_x = ang2_d1 * tmp_cost_g_q
                cost_g_x.reshape(-1, 3)[ang2_ind] += tmp_cost_g_x

                # derivative of dihed
                tmp_cost_g_q = (
                    2
                    * sin(ic.value - ic.target)
                    * sin(ang1_v) ** 2
                    * sin(ang2_v) ** 2
                    * ic.weight
                )
                tmp_cost_g_x = dihed_d1 * tmp_cost_g_q
                cost_g_x.reshape(-1, 3)[atoms] += tmp_cost_g_x

        cost_g_q = self._compute_tfm_gradient_internal()
        return np.dot(self.b_matrix.T, cost_g_q) + cost_g_x

    def _compute_tfm_hessian(self):
        """Compute tf cost function hessian in cartesian coors."""
        cost_h_x = np.zeros((3 * self.natom, 3 * self.natom))
        cost_h_q = np.zeros((len(self.ic), len(self.ic)))
        cost_g_q = self._compute_tfm_gradient_internal()
        # cost_h = np.zeros((3 * self.natom, 3 * self.natom))
        for index, ic in enumerate(self.ic):
            if isinstance(ic, (BondLength, NewDihedralCross, NewDihedralDot)):
                cost_h_q[index, index] += 2 * ic.weight
            elif isinstance(ic, BendAngle):
                cost_h_q[index, index] += (
                    2
                    * (
                        sin(ic.value) ** 2
                        - cos(ic.value) ** 2
                        + cos(ic.value) * cos(ic.target)
                    )
                    * ic.weight
                )
            elif isinstance(ic, DihedralAngle):
                atoms = np.array(ic.atoms)
                atoms_xyz = np.ravel(list(zip(atoms * 3, atoms * 3 + 1, atoms * 3 + 2)))
                assert len(atoms_xyz) == 12
                dihed_cost_h_q = np.zeros((3, 3))
                dihed_cost_g_q = np.zeros(3)
                dihed_b = np.zeros((3, 12))
                dihed_bp = np.zeros((3, 12, 12))
                # value and b matrix of angle1
                ang1_ind = atoms[:3]
                ang1_v, ang1_d1, ang1_d2 = bend_angle(self.coordinates[ang1_ind], 2)
                # value and b matrix of angle2
                ang2_ind = atoms[1:]
                ang2_v, ang2_d1, ang2_d2 = bend_angle(self.coordinates[ang2_ind], 2)

                # value of b`_ijk
                di_ind = np.array(ic.atoms)
                dihed, di_d1, di_d2 = dihed_angle(self.coordinates[di_ind], 2)
                # add dihed_b
                dihed_b[0] += di_d1.ravel()
                dihed_b[1][:9] += ang1_d1.ravel()
                dihed_b[2][3:] += ang2_d1.ravel()

                # add dihed_H
                dihed_bp[0, :, :] += di_d2.reshape(12, 12)
                dihed_bp[1, :9, :9] += ang1_d2.reshape(9, 9)
                dihed_bp[2, 3:, 3:] += ang2_d2.reshape(9, 9)

                # first derivative
                dihed_cost_g_q[0] += (
                    2 * sin(ic.value - ic.target) * sin(ang1_v) ** 2 * sin(ang2_v) ** 2
                ) * ic.weight
                dihed_cost_g_q[1] += (
                    2
                    * sin(ang1_v)
                    * cos(ang1_v)
                    * sin(ang2_v) ** 2
                    * (2 - 2 * cos(ic.value - ic.target))
                ) * ic.weight
                dihed_cost_g_q[2] += (
                    2
                    * sin(ang2_v)
                    * cos(ang2_v)
                    * sin(ang1_v) ** 2
                    * (2 - 2 * cos(ic.value - ic.target))
                ) * ic.weight

                # d2f/(d di d di)
                dihed_cost_h_q[0, 0] += (
                    2 * sin(ang1_v) ** 2 * sin(ang2_v) ** 2 * cos(ic.value - ic.target)
                ) * ic.weight
                # d2f/(da1 da1)
                dihed_cost_h_q[1, 1] += (
                    -2
                    * (-2 * cos(ic.value - ic.target) + 2)
                    * sin(ang1_v) ** 2
                    * sin(ang2_v) ** 2
                    + 2
                    * (-2 * cos(ic.value - ic.target) + 2)
                    * sin(ang2_v) ** 2
                    * cos(ang1_v) ** 2
                ) * ic.weight
                # d2f/(da2 da2)
                dihed_cost_h_q[2, 2] += (
                    -2
                    * (-2 * cos(ic.value - ic.target) + 2)
                    * sin(ang1_v) ** 2
                    * sin(ang2_v) ** 2
                    + 2
                    * (-2 * cos(ic.value - ic.target) + 2)
                    * sin(ang1_v) ** 2
                    * cos(ang2_v) ** 2
                ) * ic.weight
                # d2f/(da1 da2) and d2f/(da2 da1)
                deriv1 = (
                    4
                    * (-2 * cos(ic.value - ic.target) + 2)
                    * sin(ang1_v)
                    * sin(ang2_v)
                    * cos(ang1_v)
                    * cos(ang2_v)
                    * ic.weight
                )
                dihed_cost_h_q[1, 2] += deriv1
                dihed_cost_h_q[2, 1] += deriv1
                # d2f/(da1 ddi)
                deriv2 = (
                    4
                    * sin(ang1_v)
                    * sin(ang2_v) ** 2
                    * sin(ic.value - ic.target)
                    * cos(ang1_v)
                ) * ic.weight
                dihed_cost_h_q[1, 0] += deriv2
                dihed_cost_h_q[0, 1] += deriv2
                # d2f/(da2 ddi)
                deriv3 = (
                    4
                    * sin(ang1_v) ** 2
                    * sin(ang2_v)
                    * sin(ic.value - ic.target)
                    * cos(ang2_v)
                ) * ic.weight
                dihed_cost_h_q[0, 2] += deriv3
                dihed_cost_h_q[2, 0] += deriv3
                di_part1 = np.dot(np.dot(dihed_b.T, dihed_cost_h_q), dihed_b)
                di_part2 = np.einsum("i, ijk -> jk", dihed_cost_g_q, dihed_bp)
                dihed_cost_h_x = di_part1 + di_part2
                # print("p1", di_part1)
                # print("p2, all 0", di_part2)
                cost_h_x[atoms_xyz[:, None], atoms_xyz] += dihed_cost_h_x
        # print('costhq', cost_h_q)
        part1 = np.dot(np.dot(self.b_matrix.T, cost_h_q), self.b_matrix)
        part2 = np.einsum("i, ijk -> jk", cost_g_q, self._cc_to_ic_hessian)
        # print("haha", "!cost_h_x not 0")
        return part1 + part2 + cost_h_x

    def _compute_tfm_cost_api(self, new_coors):
        """Obeject free cost function api for scipy optimization."""
        tmp_mol = deepcopy(self)
        tmp_mol.set_new_coordinates(new_coors.reshape(-1, 3))
        return tmp_mol._compute_tfm_cost()

    def _compute_tfm_g_api(self, new_coors):
        """Obeject free cost function gradient api for scipy optimization."""
        tmp_mol = deepcopy(self)
        tmp_mol.set_new_coordinates(new_coors.reshape(-1, 3))
        return tmp_mol._compute_tfm_gradient()

    def _compute_tfm_h_api(self, new_coors):
        """Obeject free cost function hessian api for scipy optimization."""
        tmp_mol = deepcopy(self)
        tmp_mol.set_new_coordinates(new_coors.reshape(-1, 3))
        return tmp_mol._compute_tfm_hessian()

    # Deprecated method
    # def optimize_to_target_ic(
    #     self,
    #     method="BFGS",
    #     *_,
    #     max_iter=100,
    #     hess=True,
    #     hess_check=True,
    #     dihed_weight=None,
    # ):
    #     """Optimize molecule structure to given target internal coordinates.

    #     Parameters
    #     ----------
    #     method : str, default to BFGS,
    #         Optimization methods for minimize cost function. More key words
    #         can be found: https://bit.ly/2ea73op
    #     max_iter : int, default to 100
    #         Numbers of iteration for optimization loops
    #     hess : bool, False
    #         Provide analytic hessian to optimizer, mainly used to mute runtime
    #         warning for quasi-newton optimizer

    #     Raises
    #     ------
    #     NotConvergeError
    #         Raised when optimization failed
    #     """
    #     init_coor = (
    #         np.dot(pse_inv(self.b_matrix), self.target_ic - self.ic_values).reshape(
    #             -1, 3
    #         )
    #         + self.coordinates
    #     )
    #     self.set_new_coordinates(init_coor)
    #     wts_bk = None
    #     if dihed_weight is not None:
    #         wts_bk = self.ic_weights
    #         self.set_dihed_weights(dihed_weight)
    #     # for i in self.ic:
    #     #     print(i, i.weight)
    #     hess = self._compute_tfm_h_api if hess else None
    #     result = minimize(
    #         self._compute_tfm_cost_api,
    #         x0=init_coor,
    #         method=method,
    #         jac=self._compute_tfm_g_api,
    #         hess=hess,
    #         tol=1e-4,
    #         options={"maxiter": max_iter},
    #     )
    #     if wts_bk is not None:
    #         self.set_ic_weights(wts_bk)
    #     if result.success:
    #         new_coors = result.x.reshape(-1, 3)
    #         # print('opt g', self._compute_tfm_g_api(new_coors))
    #         self.set_new_coordinates(new_coors)
    #         if hess_check:
    #             hessian = self._compute_tfm_hessian()
    #             if np.any(np.linalg.eigh(hessian)[0] < -1e-3):
    #                 raise NotConvergeError(
    #                     "Converge to saddle point, need to change optimizer"
    #                 )
    #         # print("cost end", self.tf_cost)
    #     else:
    #         raise NotConvergeError("Failed to converge to target ic")

    @property
    def ic(self) -> List[CoordinateTypes]:
        """list of internal coordinates object.

        Returns
        -------
        ic : list of coordinate_types, len(ic) = K
        """
        return self._ic

    @property
    def ic_values(self) -> "np.ndarray[float]":
        """list of internal coordinates values.

        Returns
        -------
        ic_values : list of float, len(ic_values) = K
        """
        value = [i.value for i in self._ic]
        return np.array(value)

    @property
    def target_ic(self) -> "np.ndarray[float]":
        """Get target internal coordinates values.

        Returns
        -------
        target_ic : np.ndarray(K,)
        """
        target_ic = [i.target for i in self.ic]
        # if None in target_ic:  REASON: # redundant
        # raise NotSetError('Not all target ic are set')
        return np.array(target_ic)

    @property
    def connectivity(self) -> "np.ndarray[float]":
        """A connectivity matrix shows the connection of atoms.

        1 isconnected, 0 is not connected, -1 is itself

        Returns
        -------
        connectivity : np.ndarray(K, K)
        """
        return self._connectivity

    @property
    def b_matrix(self) -> "np.ndarray[float]":
        """Jacobian matrix for transforming cartisian to internal.

        Returns
        -------
        b_matrix : np.ndarray(K, 3N)
        """
        if np.any(np.isnan(self._cc_to_ic_gradient)):
            raise ValueError("NaN generated, some IC is not properly defined")
        return self._cc_to_ic_gradient

    @property
    def internal_gradient(self) -> "np.ndarray[float]":
        """Gradient of energy versus internal coordinates.

        Returns
        -------
        internal_gradient : np.ndarray(K,)
        """
        return np.dot(pse_inv(self.b_matrix.T), self.energy_gradient)

    q_gradient = internal_gradient

    @property
    def _internal_hessian(self) -> "np.ndarray[float]":
        """Hessian of energy versus internal coordinates."""
        if self._energy_hessian is None:
            return
        hes_k = self._energy_hessian - np.tensordot(
            self.internal_gradient, self._cc_to_ic_hessian, axes=1
        )
        return np.dot(np.dot(pse_inv(self.b_matrix.T), hes_k), pse_inv(self.b_matrix))

    q_hessian = _internal_hessian

    @property
    def fragments(self):
        """return a dict of fragments in the molecule."""
        unique_groups = np.unique(self._fragment)
        groups = {}
        for i in unique_groups:
            groups[i] = np.arange(self.natom)[self._fragment == i]
        return groups

    @property
    def list_ic(self):
        """Print out all internal coordiantes."""
        for index, ic in enumerate(self.ic):
            print(f"({index}), {ic}")

    def print_connectivity(self) -> None:
        """Print the connectivity matrix on screen."""
        format_func = "{:3}".format
        print("--Connectivity Starts-- \n")
        for i, _ in enumerate(self.numbers):
            print(" ".join(map(format_func, self.connectivity[i, : i + 1])))
            print("\n--Connectivity Ends--")

    def auto_select_ic(
        self,
        dihed_special: bool = False,
        reset_ic: bool = True,
        keep_bond: bool = False,
        minimum: bool = False,
    ) -> None:
        """A method for Automatically selecting internal coordinates.

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
        minimum : bool, default is False
            Select minimum amount of dihedral, one per rotatble bond.
        """
        bonds = [i for i in self.ic if isinstance(i, BondLength)]
        if reset_ic is True:
            self.wipe_ic_info(True)
        if keep_bond is False:
            self._auto_select_cov_bond()
            self._auto_select_h_bond()
            self._auto_select_fragment_bond()
            # TODO: fix auto bond function
        else:
            self.set_new_ics(bonds)
        self._auto_select_angle()
        if minimum:
            self._auto_select_minimum_dihed_normal(dihed_special)
        else:
            self._auto_select_dihed_normal(dihed_special)
            self._auto_select_dihed_improper(dihed_special)

    def _find_angle_of_atoms(self, indices):
        """find ic index for an angle consist of given indices."""
        if len(indices) != 3:
            raise ValueError(f"Input {indices} need to consist only 3 elements")
        indices = self._atoms_sequence_reorder(indices)
        for index, ic in enumerate(self.ic):
            if isinstance(ic, BendAngle) and tuple(ic.atoms) == tuple(indices):
                return index
        else:
            raise NotSetError(f"Given {indices} is not an angle in system.")

    def _delete_ic_index(self, index: int) -> None:
        """Low level function for deleting one ic."""
        del self._ic[index]
        self._regenerate_ic()
        self._regenerate_connectivity()

    def _allocate_fragment_group(self, atom1, atom2):
        """adjust fragment groups for atom1 and atom2."""
        num1 = self._fragment[atom1]
        num2 = self._fragment[atom2]
        if num1 != num2:
            self._fragment[self._fragment == num2] = num1

    def _clear_ic_info(self) -> None:  # tested
        """Wipe all the internal information in this structure."""
        self._ic = []
        self._fragment = np.arange(self.natom)
        self._connectivity = np.diag([-1] * len(self.numbers))
        self._cc_to_ic_gradient = None
        self._cc_to_ic_hessian = None

    def _auto_select_cov_bond(self):
        """Low level function for selecting covalence bond."""
        for ind1, ind2 in combinations(range(self.natom), 2):
            atom1 = self.numbers[ind1]
            atom2 = self.numbers[ind2]
            distance = self.distance(ind1, ind2)
            rad_sum = periodic[atom1].cov_radius + periodic[atom2].cov_radius
            if distance < rad_sum * 1.3:
                self.add_bond(ind1, ind2, b_type=1, weight=1)

    def _auto_select_h_bond(self):
        """Low level function for selecting hydrogen bond."""
        ele_neg = np.array([7, 8, 9, 15, 16, 17])
        # find strong ele nagative atoms' indices
        halo_indices = np.where(np.isin(self.numbers, ele_neg))[0]
        for ha_idx in halo_indices:
            # indices formed cov bond with index ha_idx
            cnnt_idxs = np.where(self.connectivity[ha_idx] == 1)[0]
            # indices of H atoms formed bond with ha_idx
            cnnt_h_idx = cnnt_idxs[self.numbers[cnnt_idxs] == 1]
            for h_idx in cnnt_h_idx:
                # loop over second halo atoms
                for ha_idx2 in halo_indices:
                    if ha_idx2 != ha_idx and self.connectivity[h_idx][ha_idx2] == 0:
                        dist = self.distance(h_idx, ha_idx2)
                        angle_cos = self.angle_cos(ha_idx, h_idx, ha_idx2)
                        cut_off = (
                            periodic[self.numbers[h_idx]].vdw_radius
                            + periodic[self.numbers[ha_idx2]].vdw_radius
                        )
                        if dist <= 0.9 * cut_off and angle_cos < 0:
                            self.add_bond(h_idx, ha_idx2, b_type=2, weight=1)

    def _auto_select_fragment_bond(self):
        """automatically select fragmental bonds."""
        frags = self.fragments
        # print(frags.keys())
        for group1, group2 in combinations(frags.keys(), 2):
            atoms_set = []
            # atom indices for each fragments
            g1_atoms = frags[group1]  # np.array
            g2_atoms = frags[group2]  # np.array

            # atomic number for each fragments
            g1_numbers = self.numbers[g1_atoms]
            g2_numbers = self.numbers[g2_atoms]

            # min atoms for each fragments
            # min_atom = min(len(g1_atoms), len(g2_atoms))
            # most non h atoms
            max_f_bond = max(np.sum([g1_numbers != 1]), np.sum([g2_numbers != 1]))
            for atom_1 in g1_atoms:
                # print('a1', atom_1)
                for atom_2 in g2_atoms:
                    # print('a2', atom_2)
                    new_distance = self.distance(atom_1, atom_2)
                    if len(atoms_set) > 1 and new_distance > 2 * atoms_set[0][0]:
                        continue
                    heappush(atoms_set, (new_distance, atom_1, atom_2))

            counter = 0
            least_length = atoms_set[0][0]
            while atoms_set:
                bond_dis, atom1, atom2 = heappop(atoms_set)
                if counter < 2:
                    self.add_bond(atom1, atom2, b_type=3, weight=1)
                    counter += 1
                    continue
                if bond_dis > max(3.7794520, 2 * least_length):
                    break
                if counter >= max(2, max_f_bond):
                    break
                self.add_bond(atom1, atom2, b_type=3, weight=1)
                counter += 1

    def _auto_select_angle(self) -> None:
        """A private method for automatically selecting angle."""
        for center_index, _ in enumerate(self.numbers):
            connected = self.connected_indices(center_index, exclude=4)
            # connected = self.connected_indices(center_index)
            if len(connected) >= 2:
                for side_1, side_2 in combinations(connected, 2):
                    self.add_angle(side_1, center_index, side_2, weight=1)

    def _auto_select_dihed_normal(self, special=False) -> None:
        """A private method for automatically selecting normal dihedral."""
        for center_ind_1, _ in enumerate(self.numbers):
            # find indices connected to center_ind_1
            connected = self.connected_indices(center_ind_1, exclude=4)
            if len(connected) >= 2:
                for center_ind_2 in connected:
                    # TODO: may introduce bugs in the future
                    sum_cnct = np.sum(self.connectivity, axis=0)
                    # find total connection for all atoms connected c1
                    sum_select_cnct = sum_cnct[connected]
                    sorted_index = sum_select_cnct.argsort()[::-1]
                    side_1 = connected[sorted_index[0]]
                    # select the atom with the largest connection
                    if connected[sorted_index[0]] == center_ind_2:
                        side_1 = connected[sorted_index[1]]
                    connected_to_index_2 = self.connected_indices(
                        center_ind_2, exclude=4
                    )
                    for side_2 in connected_to_index_2:
                        if side_2 not in (center_ind_1, center_ind_2, side_1):
                            self.add_dihedral(
                                side_1,
                                center_ind_1,
                                center_ind_2,
                                side_2,
                                special=special,
                            )

    def _auto_select_minimum_dihed_normal(self, special=False):
        """Low level code for select one dihedral per rotatable bond."""
        for i in range(self.natom - 1):
            for j in range(i + 1, self.natom):
                if self.connectivity[i][j] <= 0 or self.connectivity[i][j] == 4:
                    continue
                cnct_to_i = self.connected_indices(i, exclude=4)
                cnct_to_j = self.connected_indices(j, exclude=4)
                if len(cnct_to_i) < 2 or len(cnct_to_j) < 2:
                    continue
                if len(set(cnct_to_i).union(set(cnct_to_j))) < 4:
                    continue
                sorted_index_i = self.number_of_connect[cnct_to_i].argsort()[::-1]
                for k in sorted_index_i:
                    if cnct_to_i[k] != j:
                        side_i = cnct_to_i[k]
                        break
                sorted_index_j = self.number_of_connect[cnct_to_j].argsort()[::-1]
                for k in sorted_index_j:
                    if cnct_to_j[k] not in (side_i, i):
                        side_j = cnct_to_j[k]
                        break
                self.add_dihedral(side_i, i, j, side_j, special=special)

    def _auto_select_dihed_improper(self, special=False) -> None:
        """A private method for automatically selecting improper dihedral."""
        connect_sum = np.sum(self.connectivity > 0, axis=0)
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
                        self.add_dihedral(
                            ind_i, center_ind, ind_j, ind_k, special=special
                        )

    def _regenerate_ic(self) -> None:
        """Reset internal coordinates system.

        Reset gradient, hessian versus internal coordinates, regenerate
        internal coordinates and transformation matrix.
        """
        self._cc_to_ic_gradient = None
        self._cc_to_ic_hessian = None
        for ic in self.ic:
            rs = self.coordinates[np.array(ic.atoms)]
            ic.set_new_coordinates(rs)
            d, dd = ic.get_gradient_hessian()
            self._add_cc_to_ic_gradient(d, ic.atoms)  # add transform gradient
            self._add_cc_to_ic_hessian(dd, ic.atoms)  # add transform hessian

    def _regenerate_connectivity(self) -> None:
        """regenerate the connectivity of molecule depends on present
        internal coordinates
        """
        self._connectivity = np.diag([-1] * len(self.numbers))
        for ic in self.ic:
            if isinstance(ic, BondLength):
                self._add_connectivity(ic.atoms, ic.ic_type)

    def create_geo_point(self, flex_sin=True) -> Point:
        """Create a Point obj. in IC to for transformation optimization.

        Returns
        -------
        geo_point : Point object
        """
        # _, x_d, x_dd = self.cost_value_in_cc
        # same for special dihed. Works for conventional
        if flex_sin is True:
            x_d = self._compute_tfm_gradient()
            x_dd = self._compute_tfm_hessian()
        else:
            # not always converge but better result
            _, x_d, x_dd = self.cost_value_in_cc
        return Point(x_d, x_dd, len(self.numbers))

    def _ic_gradient_hessian_transform_to_cc(
        self, gradient: "np.ndarray[float]", hessian: "np.ndarray[float]"
    ) -> Tuple["np.ndarray[float]", "np.ndarray[float]"]:
        """Transform energy gradient and hessian  from internal coordinates to cartesian.
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
        # cartesian_gradient = np.dot(gradient, self.b_matrix)
        cartesian_gradient = np.dot(self.b_matrix.T, gradient)
        cartesian_hessian_part_1 = np.dot(
            np.dot(self.b_matrix.T, hessian), self.b_matrix
        )
        cartesian_hessian_part_2 = np.tensordot(gradient, self._cc_to_ic_hessian, 1)
        cartesian_hessian = cartesian_hessian_part_1 + cartesian_hessian_part_2
        return cartesian_gradient, cartesian_hessian

    def _change_weight(self, index, value):
        """change weight of given index internal coordinates to given value"""
        self.ic[index].weight = value

    def _check_connectivity(self, atom1: int, atom2: int) -> bool:
        """Check whether two atoms are connected or not.

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
        if self.connectivity[atom1, atom2] > 0:
            return True
        elif self.connectivity[atom1, atom2] == 0:
            return False

    def _repeat_atoms_check(self, atoms) -> bool:
        """Check whether the given atoms already existed in ic atoms or not.

        Arguments
        ---------
        atoms : tuple
            the given ic object to be tested

        Returns
        -------
        bool
            Return True if there is no duplicate and it's a valid new ic
            object, otherwise False
        """
        if len(atoms) <= 3:
            for ic in self.ic:
                if atoms == ic.atoms:
                    return False
            return True
        if len(atoms) == 4:
            for ic in self.ic:
                if len(ic.atoms) == 4:
                    mid = atoms[1:3]
                    side = atoms[0], atoms[3]
                    mid_ref = ic.atoms[1:3]
                    side_ref = ic.atoms[0], ic.atoms[3]
                    if all(np.sort(mid) == np.sort(mid_ref)) and all(
                        np.sort(side) == np.sort(side_ref)
                    ):
                        return False
            return True

    def _add_new_internal_coordinate(
        self,
        new_ic: CoordinateTypes,
        d: "np.ndarray[float]",
        dd: "np.ndarray[float]",
        atoms: Tuple[int, ...],
    ) -> None:
        """Add a new ic object to the system and corresponding transformation parts."""
        self._ic.append(new_ic)
        self._add_cc_to_ic_gradient(d, atoms)  # add gradient
        self._add_cc_to_ic_hessian(dd, atoms)  # add hessian

    def _add_connectivity(self, atoms: Tuple[int, ...], bond_type) -> None:
        """Change the value of connectivity matrix to bond_type(int) for two atoms.

        Arguments
        ---------
        atoms : list or tuple of int, len(atoms) = 2
        """
        if len(atoms) != 2:
            raise AtomsNumberError("The number of atoms is not correct")
        num1, num2 = atoms
        self._connectivity[num1, num2] = bond_type
        self._connectivity[num2, num1] = bond_type

    @staticmethod
    def _atoms_sequence_reorder(atoms: Tuple[int, ...]) -> None:
        """Change the atoms in each ic object in ascending sequence.

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
            pass
        else:
            raise AtomsNumberError("The number of atoms is not correct")
        return tuple(atoms)

    def _add_cc_to_ic_gradient(
        self, deriv: "np.ndarray[float]", atoms: Tuple[int, ...]
    ) -> None:  # need to be tested
        """Add new entries from a new ic to transformation matrix gradient.

        Arguments
        ---------
        deriv : np.ndarray(3N,)
            tranformation gradient matrix given ic regarding to given atoms
        atoms : list or tuple of int
            indices of atoms for those transformation
        """
        if self._cc_to_ic_gradient is None:
            self._cc_to_ic_gradient = np.zeros((0, 3 * len(self.numbers)))
        tmp_vector = np.zeros(3 * len(self.numbers))
        atoms = np.array(atoms)
        index = np.ravel(list(zip(3 * atoms, 3 * atoms + 1, 3 * atoms + 2)))
        tmp_vector[index] += deriv.ravel()
        self._cc_to_ic_gradient = np.vstack((self._cc_to_ic_gradient, tmp_vector))

    def _add_cc_to_ic_hessian(
        self, deriv: "np.ndarray[float]", atoms: Tuple[int, ...]
    ) -> None:  # need to be tested
        """Add new entries from a new ic to transformation matrix hessian.

        Arguments
        ---------
        deriv : np.ndarray(3N, 3N)
            tranformation hessian matrix given ic regarding to given atoms
        atoms : list or tuple of int
            indices of atoms for those transformation
        """
        if self._cc_to_ic_hessian is None:
            self._cc_to_ic_hessian = np.zeros(
                (0, 3 * len(self.numbers), 3 * len(self.numbers))
            )
        atoms = np.array(atoms)
        tmp_vector = np.zeros((1, 3 * len(self.numbers), 3 * len(self.numbers)))
        index = np.ravel(list(zip(3 * atoms, 3 * atoms + 1, 3 * atoms + 2)))
        tmp_vector[0, index[:, None], index] = deriv.reshape(-1, 3 * len(atoms))
        self._cc_to_ic_hessian = np.vstack((self._cc_to_ic_hessian, tmp_vector))

    @staticmethod
    def _direct_square(origin: float, target: float) -> Tuple[float, ...]:
        """Calculate cost function and it's derivatives for geometry transiformation.

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
        value = (origin - target) ** 2
        deriv = 2 * (origin - target)
        deriv2 = 2
        return value, deriv, deriv2
