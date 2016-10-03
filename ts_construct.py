from __future__ import print_function, absolute_import
from saddle.internal import Internal
from saddle.errors import AtomsNumberError, InputTypeError
from copy import deepcopy
from horton import periodic
import numpy as np


class TSConstruct(object):

    def __init__(self, reactant_ic, product_ic):
        if isinstance(reactant_ic, Internal) and isinstance(product_ic, Internal):
            if reactant_ic.numbers.all() == product_ic.numbers.all():
                self._numbers = reactant_ic.numbers
                self._reactant = reactant_ic
                self._product = product_ic
            else:
                raise AtomsNumberError("The number of atoms is not the same")
        else:
            raise InputTypeError("The type of input data is illegal.")

    @property
    def reactant(self):
        return self._reactant

    @property
    def product(self):
        return self._product

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
        pass

    def _auto_select_bond(self):
        halidish_atom = set([7, 8, 9, 15, 16, 17])
        for index_i in range(len(self._numbers)):
            for index_j in range(index_i + 1, len(self._numbers)):
                atom_num1 = self.numbers[index_i]
                atom_num2 = self.numbers[index_j]
                distance_rct = self._reactant.distance(index_i, index_j)
                distance_prd = self._product.distance(index_i, index_j)
                radius_sum = periodic[
                    atom_num1].cov_radius + periodic[atom_num2].cov_radius
                if distance_prd < 1.3 * radius_sum or distance_rct < 1.3 * radius_sum:
                    self.add_bond(index_i, index_j)
                    if min(atom_num1, atom_num2) == 1 and max(atom_num1, atom_num2) in halidish_atom:
                        if atom_num1 == 1:
                            h_index = index_i
                            halo_index = index_j
                        else:
                            h_index = index_j
                            halo_index = index_i
                        for index_k in range(index_j + 1, len(self._numbers)):
                            atom_num3 = self.number[index_k]
                            if atom_num3 in halidish_atom:
                                dis_r = self._reactant.distance(
                                    h_index, index_k)
                                dis_p = self._product.distance(
                                    h_index, index_k)
                                angle_r = self._reactant.angle(
                                    halo_index, h_index, index_k)
                                angle_p = self._product.angle(
                                    halo_index, h_index, index_k)
                                thresh_sum = periodic[self._numbers[
                                    h_index]].vdw_radius + periodic[self._numbers[index_k]].vdw_radius
                                if min(dis_r, dis_p) <= 0.9 * thresh_sum and min(angle_p, angle_r) <= 0.:
                                    self.add_bond(h_index, index_k)
        # did't add aux bond method

    def _auto_select_angle(self):
        for center_index in range(len(self._numbers)):
            connection = self._reactant.connectivity[center_index]
            connected = connection[connection > 0]
            if len(connected) >= 2:
                for edge_1 in range(len(connected)):
                    for edge_2 in range(edge_1 + 1, len(connected)):
                        self.add_angle_cos(edge_1, center_index, edge_2)

    def _auto_select_dihed_normal(self):
        for center_index_1 in range(len(self._numbers)):
            connection = self._reactant.connectivity[center_index_1]
            connected_to_index_1 = connection[connection > 0]
            for center_index_2 in connected_to_index_1:
                maximum_connect = np.sum(self._reactant.connectivity, axis=0)
                mask = np.ones(len(maximum_connect), dtype=bool)
                mask[[maximum_connect]] = False
                mask[center_index_2] = False
                maximum_cnct = maximum_connect[mask]
                max_index = maximum_cnct[maximum_cnct == maximum_cnct.max()][0]
                connection = self._reactant.connectivity[center_index_2]
                connected_to_index_2 = connection[connection > 1]
                for side_2 in connected_to_index_2:
                    self.add_dihedral(max_index, center_index_1,
                                      center_index_2, side_2)
