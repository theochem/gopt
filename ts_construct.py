from __future__ import absolute_import, print_function

from copy import deepcopy

import numpy as np
from horton import periodic

from saddle.errors import AtomsNumberError, InputTypeError, NotSetError
from saddle.internal import Internal


class TSConstruct(object):
    def __init__(self, reactant_ic, product_ic):
        if isinstance(reactant_ic, Internal) and isinstance(product_ic,
                                                            Internal):
            if np.allclose(reactant_ic.numbers, product_ic.numbers):
                self._numbers = reactant_ic.numbers
                self._reactant = reactant_ic
                self._product = product_ic
            else:
                raise AtomsNumberError("The number of atoms is not the same")
        else:
            raise InputTypeError("The type of input data is illegal.")
        self._key_ic_counter = 0
        self._ts = None

    @property
    def reactant(self):
        return self._reactant

    @property
    def product(self):
        return self._product

    @property
    def ts(self):
        if self._ts is None:
            raise NotSetError("TS state hasn't been set")
        return self._ts

    @property
    def numbers(self):
        return self._numbers

    @property
    def key_ic_counter(self):
        return self._key_ic_counter

    def add_bond(self, atom1, atom2):
        self._reactant.add_bond(atom1, atom2)
        self._product.add_bond(atom1, atom2)

    def add_angle_cos(self, atom1, atom2, atom3):
        self._reactant.add_angle_cos(atom1, atom2, atom3)
        self._product.add_angle_cos(atom1, atom2, atom3)

    def add_dihedral(self, atom1, atom2, atom3, atom4):
        self._reactant.add_dihedral(atom1, atom2, atom3, atom4)
        self._product.add_dihedral(atom1, atom2, atom3, atom4)

    def auto_select_ic(self):
        self._auto_select_bond()
        self._auto_select_angle()
        self._auto_select_dihed_normal()
        self._auto_select_dihed_improper()

    def create_ts_state(self, start_with, ratio=0.5):
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
        ts_internal.converge_to_target_ic(100)
        self._ts = ts_internal  # set _ts attribute

    def select_key_ic(self, ic_index):
        if ic_index < self._key_ic_counter:
            # if the index is smaller then ic counter, it is pointless to swap
            return
        self._ts.swap_internal_coordinates(self._key_ic_counter, ic_index)
        self._key_ic_counter += 1

    def _auto_select_bond(self):
        halidish_atom = set([7, 8, 9, 15, 16, 17])
        for index_i in range(len(self.numbers)):
            for index_j in range(index_i + 1, len(self.numbers)):
                atom_num1 = self.numbers[index_i]
                atom_num2 = self.numbers[index_j]
                distance_rct = self._reactant.distance(index_i, index_j)
                distance_prd = self._product.distance(index_i, index_j)
                radius_sum = periodic[atom_num1].cov_radius + periodic[
                    atom_num2].cov_radius
                if min(distance_prd, distance_rct) < 1.3 * radius_sum:
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
                            atom_num3 = self.number[index_k]
                            if atom_num3 in halidish_atom:
                                dis_r = self._reactant.distance(h_index,
                                                                index_k)
                                dis_p = self._product.distance(h_index,
                                                               index_k)
                                angle_r = self._reactant.angle(
                                    halo_index, h_index, index_k)
                                angle_p = self._product.angle(halo_index,
                                                              h_index, index_k)
                                thresh_sum = periodic[self._numbers[
                                    h_index]].vdw_radius + \
                                    periodic[self._numbers[index_k]].vdw_radius
                                if (min(dis_r, dis_p) <= 0.9 * thresh_sum and
                                        max(angle_p, angle_r) >= 1.57079632):
                                    self.add_bond(h_index, index_k)
        # did't add aux bond method

    def _auto_select_angle(self):
        for center_index in range(len(self.numbers)):
            connected = self._reactant.connected_indices(center_index)
            if len(connected) >= 2:
                for edge_1 in range(len(connected)):
                    for edge_2 in range(edge_1 + 1, len(connected)):
                        self.add_angle_cos(connected[edge_1], center_index,
                                           connected[edge_2])

    def _auto_select_dihed_normal(self):
        for center_ind_1 in range(len(self.numbers)):
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
        connect_sum = np.sum(self._reactant.connectivity, axis=0)
        for center_ind in range(len(connect_sum)):
            if connect_sum[center_ind] >= 3:
                cnct_atoms = self._reactant.connected_indices(center_ind)
                cnct_total = len(cnct_atoms)
                for i in range(cnct_total):
                    for j in range(i + 1, cnct_total):
                        for k in range(j + 1, cnct_total):
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
