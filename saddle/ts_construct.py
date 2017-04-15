from __future__ import absolute_import, print_function

from copy import deepcopy
from itertools import combinations

import numpy as np

from .errors import AtomsNumberError, InputTypeError, NotSetError
from .internal import Internal
from .periodic import periodic
from .reduced_internal import ReducedInternal

__all__ = ['TSConstruct']


class TSConstruct(object):
    """Transitian State Constructor

    Properties
    ----------
    reactant : Internal
        internal coordinates structure of reactant of certain reaction
    product : Internal
        internal coordinates structure of product of certain reaction
    ts : Internal
        internal coordinates structure of initial transition state guess of
        certain chemical reaction
    numbers : np.ndarray(N,)
        A numpy array of atomic number for input coordinates
    key_ic_number : int
        Number of key internal coordinates which correspond to important
        chemical property

    Methods
    -------
    __init__(reactant_ic, product_ic)
        Initializes constructor with the input of two Internal instance,
        each represent the structure of reactant and product respectively
    add_bond(atom1, atom2)
        Add bond connection for both reactant and product structures
        between atoms1 and atoms2
    add_angle_cos(atom1, atom2, atom3)
        Add angle cos to both reactant and product between angle atom1,
        atom2, and atom3
    add_dihedral(atom1, atom2, atom3)
        Add normal dihedral to both reactant and product between plane
        (atom1, atom2, and atom3) and plane(atom2, atom3, and atom4)
    auto_select_ic()
        Auto select internal coordinates of reactant and product with
        specific choices
    create_ts_state(start_with, ratio=0.5)
        Create transition state based on reactant and product internal
        coordinates
    select_key_ic(ic_indices)
        Select certain internal coordinates to be the key ic which is
        important to chemical reaction
    auto_generate_ts(ratio=0.5, start_with="reactant", reconstruct=True)
        Generate transition state structure automatically based on the reactant
        and product structure and corresponding setting parameters
    """

    def __init__(self, reactant_ic, product_ic):
        if isinstance(reactant_ic, Internal) and isinstance(product_ic,
                                                            Internal):
            if np.allclose(reactant_ic.numbers, product_ic.numbers):
                self._numbers = reactant_ic.numbers
                self._reactant = deepcopy(reactant_ic)
                self._reactant.wipe_ic_info(True)
                self._product = deepcopy(product_ic)
                self._product.wipe_ic_info(True)
                self._tmp_rct_ic = reactant_ic.ic  # the ref to reactant_ic
                self._tmp_prd_ic = product_ic.ic  # the ref to product_ic
            else:
                raise AtomsNumberError("The number of atoms is not the same")
        else:
            raise InputTypeError("The type of input data is invalid.")
        self._key_ic_counter = 0
        self._ts = None

    @property
    def reactant(self):
        """Internal coordinates instance of reactant structure

        Returns
        -------
        reactant : Internal
        """
        return self._reactant

    @property
    def product(self):
        """Internal coordinates instance of product structure

        Returns
        -------
        product : Internal
        """
        return self._product

    @property
    def ts(self):
        """Internal cooridnates instance of transition structure if
        it has been generated already. Otherwise, raise NotSetError.

        Returns
        -------
        ts : Internal
        """
        if self._ts is None:
            raise NotSetError("TS state hasn't been set")
        return self._ts

    @property
    def numbers(self):
        """A numpy array of atomic number for input coordinates

        Returns
        -------
        numbers : np.ndarray(N,)
        """
        return self._numbers

    @property
    def key_ic_counter(self):
        """Number of key internal coordinates in this reaction

        Returns
        -------
        key_ic_counter : int
        """
        return self._key_ic_counter

    def add_bond(self, atom1, atom2):
        """Add bond connection between atom1 and atom2 for both reactant
        and product structure

        Arguments
        ---------
        atom1 : int
            The index of the first atom of a bond
        atom2 : int
            The index of the second atom of a bond
        """
        self._reactant.add_bond(atom1, atom2)
        self._product.add_bond(atom1, atom2)

    def add_angle_cos(self, atom1, atom2, atom3):
        """Add cos angle connection between atom1, atom2, and atom3 for
        both reactant and product structure

        Arguments
        ---------
        atom1 : int
            The index of the first atom of the angle
        atom2 : int
            The index of the second atom of the angle
        atom3 : int
            The index of the third atom of the angle
        """
        self._reactant.add_angle_cos(atom1, atom2, atom3)
        self._product.add_angle_cos(atom1, atom2, atom3)

    def add_dihedral(self, atom1, atom2, atom3, atom4):
        """Add dihedral angle between plane1(atom1, atom2, and atom3)
        and plane2(atom2, atom3, and atom4) for both reactant and
        product structures

        Arguments
        ---------
        atom1 : int
            The index of atom1 in plane1
        atom2 : int
            The index of atom2 in plane1 and plane2
        atom3 : int
            The index of atom3 in plane1 and plane2
        atom4 : int
            The index of atom4 in plane2
        """
        self._reactant.add_dihedral(atom1, atom2, atom3, atom4)
        self._product.add_dihedral(atom1, atom2, atom3, atom4)

    def auto_select_ic(self, reconstruct=True):
        """automatically select internal coordinates based on the internal
        structures of reactant and product

        Arguments
        ---------
        reconstruct : bool, default is True
            The flag of whether to construct the internal structure of reactant
            and product from scratch. True for start from scrach, otherwise False.
        """
        self._reactant.wipe_ic_info(True)
        self._product.wipe_ic_info(True)
        if reconstruct:
            self._auto_select_ic_restart()
        else:
            self._auto_select_ic_combine()

    def create_ts_state(self, start_with, ratio=0.5):
        """Create transition state structure based on the linear combination of
        internal structure of both reactant and product.

        Arguments
        ---------
        start_with : string
            The initial structure of transition state to optimize from.
        ratio : float, default is 0.5
            The ratio of linear combination of ic for reactant and product.
            ts = ratio * reactant + (1 - ratio) * product
        """
        if start_with == "reactant":
            model = self.reactant
        elif start_with == "product":
            model = self.product
        else:
            raise InputTypeError("The input of start_with is not supported")
        if ratio > 1. or ratio < 0:
            raise InputTypeError("The input of ratio is not supported")
        ts_internal = deepcopy(model)
        target_ic = ratio * self.reactant.ic_values + (
            1. - ratio) * self.product.ic_values
        ts_internal.set_target_ic(target_ic)
        ts_internal.converge_to_target_ic()
        ts_internal = ReducedInternal.update_to_reduced_internal(ts_internal)
        # change the ts_internal to Class ReducedInternal
        self._ts = ts_internal  # set _ts attribute

    def select_key_ic(self, *ic_indices):
        """Set one or multiply internal coordinate(s) as the the key internal
        coordinates

        Arguments
        ---------
        *ic_indices : *int
            the index(indices) of internal coordinates
        """
        for index in ic_indices:
            if index < self._key_ic_counter:
                # if the index is smaller then ic counter, it is pointless to swap
                continue
            self.ts.swap_internal_coordinates(self._key_ic_counter, index)
            self._key_ic_counter += 1
            self.ts.set_key_ic_number(self._key_ic_counter)

    def auto_generate_ts(self,
                         ratio=0.5,
                         start_with="reactant",
                         reconstruct=True):
        """Complete auto generate transition state structure based on some
        default parameters

        Arguments
        ---------
        ratio : float, default is 0.5
            The ratio of linear combination of ic for reactant and product.
            ts = ratio * reactant + (1 - ratio) * product
        start_with : string, default is "reactant"
            The initial structure of transition state to optimize from.
        reconstruct : bool, default is True
            The flag of whether to construct the internal structure of reactant
            and product from scratch. True for start from scrach, otherwise False.
        """
        self.auto_select_ic(reconstruct)
        self.create_ts_state(start_with, ratio)

    def _auto_select_ic_restart(self):
        """Based on buildin to auto-select algorithm to select internal
        coordinates from scrach. Do not include any initial structure of
        reactant or product
        """
        self._auto_select_bond()
        self._auto_select_angle()
        self._auto_select_dihed_normal()
        self._auto_select_dihed_improper()

    def _auto_select_ic_combine(self):
        """Based the internal structure of both reactant and product,
        combine the structure to form a unified structure.
        """
        # obetain ic for both reactant and product
        union_ic_list = self._get_union_of_ics()
        self._reactant.set_new_ics(union_ic_list)
        self._product.set_new_ics(union_ic_list)

    def _get_union_of_ics(self):  # need tests
        """Get the combined internal coordinates based on the ic structure of
        both reactant and product
        """
        basic_ic = deepcopy(self._tmp_rct_ic)
        for new_ic in self._tmp_prd_ic:
            for ic in self._tmp_rct_ic:
                if new_ic.atoms == ic.atoms and type(new_ic) == type(ic):
                    break
            else:
                basic_ic.append(new_ic)
        return basic_ic

    def _auto_select_bond(self):
        """Automatically select bond connection based on buildin algorithm
        for both reactant and product
        """
        halidish_atom = set([7, 8, 9, 15, 16, 17])
        all_halo_index = (i for i, j in enumerate(self.numbers)
                          if j in halidish_atom)
        for index_i, index_j in combinations(range(len(self.numbers)), 2):
            atom_num1 = self.numbers[index_i]
            atom_num2 = self.numbers[index_j]
            distance_rct = self._reactant.distance(index_i, index_j)
            distance_prd = self._product.distance(index_i, index_j)
            radius_sum = periodic[atom_num1].cov_radius + periodic[
                atom_num2].cov_radius
            if min(distance_prd, distance_rct) < 1.3 * radius_sum:
                self.add_bond(index_i, index_j)
                if atom_num1 == 1 and atom_num2 in halidish_atom:
                    h_index = index_i
                    halo_index = index_j
                elif atom_num2 == 1 and atom_num1 in halidish_atom:
                    h_index = index_j
                    halo_index = index_i
                else:
                    continue
                potent_halo_index = (i for i in all_halo_index
                                     if i != halo_index)
                for index_k in potent_halo_index:
                    dis_r = self._reactant.distance(h_index, index_k)
                    dis_p = self._product.distance(h_index, index_k)
                    angle_r = self._reactant.angle(halo_index, h_index,
                                                   index_k)
                    angle_p = self._product.angle(halo_index, h_index, index_k)
                    thresh_sum = periodic[self._numbers[
                        h_index]].vdw_radius + periodic[self._numbers[
                            index_k]].vdw_radius
                    if (min(dis_r, dis_p) <= 0.9 * thresh_sum and
                            max(angle_p, angle_r) >= 1.57079632):
                        self.add_bond(h_index, index_k)
                # did't add aux bond method

    def _auto_select_angle(self):
        """Automatically select angle based on buildin algorithm for both
        reactant and product
        """
        for center_index, _ in enumerate(self.numbers):
            connected = self._reactant.connected_indices(center_index)
            if len(connected) >= 2:
                for side_1, side_2 in combinations(connected, 2):
                    self.add_angle_cos(side_1, center_index, side_2)

    def _auto_select_dihed_normal(self):
        """Automatically select dihedral based on buildin algorithm for both
        reactant and product
        """
        for center_ind_1, _ in enumerate(self.numbers):
            connected = self._reactant.connected_indices(center_ind_1)
            if len(connected) >= 2:
                for center_ind_2 in connected:
                    sum_cnct = np.sum(self._reactant.connectivity, axis=0)
                    sum_select_cnct = sum_cnct[connected]
                    sorted_index = sum_select_cnct.argsort()[::-1]
                    side_1 = connected[sorted_index[0]]
                    if connected[sorted_index[0]] == center_ind_2:
                        side_1 = connected[sorted_index[1]]
                    connected_to_index_2 = self._reactant.connected_indices(
                        center_ind_2)
                    for side_2 in connected_to_index_2:
                        if side_2 not in (center_ind_1, center_ind_2, side_1):
                            self.add_dihedral(side_1, center_ind_1,
                                              center_ind_2, side_2)

    def _auto_select_dihed_improper(self):
        """Automatically select improper dihedral based on buildin algorithm for both
        reactant and product
        """
        connect_sum = np.sum(self._reactant.connectivity, axis=0)
        for center_ind, _ in enumerate(connect_sum):
            if connect_sum[center_ind] >= 3:
                cnct_atoms = self._reactant.connected_indices(center_ind)
                cnct_total = len(cnct_atoms)
                for i, j, k in combinations(range(cnct_total), 3):
                    ind_i, ind_j, ind_k = cnct_atoms[[i, j, k]]
                    ang1_r = self._reactant.angle(ind_i, center_ind,
                                                  ind_j)
                    ang2_r = self._reactant.angle(ind_i, center_ind,
                                                  ind_k)
                    ang3_r = self._reactant.angle(ind_j, center_ind,
                                                  ind_k)
                    ang1_p = self._product.angle(ind_i, center_ind,
                                                 ind_j)
                    ang2_p = self._product.angle(ind_i, center_ind,
                                                 ind_k)
                    ang3_p = self._product.angle(ind_j, center_ind,
                                                 ind_k)
                    sum_r = ang1_r + ang2_r + ang3_r
                    sum_p = ang1_p + ang2_p + ang3_p
                    if max(sum_p, sum_r) >= 6.02139:
                        self.add_dihedral(ind_i, center_ind, ind_j,
                                          ind_k)
