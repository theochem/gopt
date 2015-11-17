# import numpy as np
import horton as ht

from horton.periodic import periodic
from saddle.ICTransformation import ICTransformation


class TransitionSearch(object):

    def __init__(self, reagent, product):
        self.reagent = ICTransformation(reagent.coordinates)
        self.product = ICTransformation(product.coordinates)
        if reagent.numbers.tolist() != product.numbers.tolist():
            raise AtomsNumberError
        self.numbers = reagent.numbers
        self.len = len(self.numbers)
        self.halo_atom_index = set()
        for i in range(self.len):
            if self.numbers[i] in TransitionSearch.halo_atom_numbers:
                self.halo_atom_index.add(i)
        self.ts_state = None
        self._ic_key_counter = 0

    halo_atom_numbers = (7, 8, 9, 15, 16, 17)

    def auto_ts_search(self, ratio=0.5):
        self.auto_ic_select(self.reagent,[self.reagent, self.product])
        self.get_ts_guess_cc(ratio)
        self.get_ts_guess_ic(ratio)

    def get_ts_guess_cc(self, ratio=0.5):
        if ratio > 1. or ratio < 0.:
            raise ValueError
        ts_coordinate = self.reagent.coordinates * \
            ratio + self.product.coordinates * (1. - ratio)
        self.ts_state = ICTransformation(ts_coordinate)

    def get_ts_guess_ic(self, ratio=0.5):
        if len(self.reagent.ic) != len(self.product.ic):
            raise AtomsNumberError
        if ratio > 1. or ratio < 0.:
            raise ValueError
        target_ic = self.reagent.ic * ratio + self.product.ic * (1. - ratio)
        self.ts_state.target_ic = target_ic

    @staticmethod
    def add_bond(atom1, atom2, b_type, multistructure):
        for structure in multistructure:
            structure.add_bond_length(atom1, atom2, b_type)

    @staticmethod
    def add_angle(atom1, atom2, atom3, multistructure):
        for structure in multistructure:
            structure.add_bend_angle(atom1, atom2, atom3)

    @staticmethod
    def add_dihed_conv(atom1, atom2, atom3, atom4, multistructure):
        for structure in multistructure:
            structure.add_dihed_angle(atom1, atom2, atom3, atom4)

    @staticmethod
    def add_dihed_new(atom1, atom2, atom3, atom4, multistructure):
        for structure in multistructure:
            structure.add_dihed_new(atom1, atom2, atom3, atom4)

    @staticmethod
    def add_aux_bond(atom1, atom2, multistructure):
        for structure in multistructure:
            structure.add_aux_bond(atom1, atom2)

    @staticmethod
    def upgrade_aux_bond(atom1, atom2, multistructure):
        for structure in multistructure:
            structure.upgrade_aux_bond(atom1, atom2)

    def auto_ic_select(self, selected_structure, target_structure=[]):
        assert isinstance(target_structure,
                          list), "target_structure should be a list"
        if not target_structure:
            target_structure.append(selected_structure)
        self._auto_bond_select(selected_structure, target_structure)
        self._auto_angle_select(selected_structure, target_structure)
        self._auto_dihed_select(selected_structure, target_structure)

    def _auto_bond_select(self, selected, targeted):
        # i,j is index of selected atoms in coordinates
        for index_i in range(self.len):
            for index_j in range(index_i + 1, self.len):
                atomnum1 = self.numbers[index_i]
                atomnum2 = self.numbers[index_j]
                # atomnum1, atomnum2 is atomic number of two selected atom2
                distance = selected.length_calculate(index_i, index_j)
                radius_sum = periodic[
                    atomnum1].cov_radius + periodic[atomnum2].cov_radius
                if distance < 1.3 * radius_sum:
                    self.add_bond(
                        index_i, index_j, "covalence", targeted)
                    result = self._hydrogen_halo_test(
                        index_i, index_j)  # do hydrogen bond exam
                    if result[0]:
                        for index3 in self.halo_atom_index:
                            index1, index2 = result[1]
                            atomnum1, atomnum2 = self.numbers[
                                index1], self.numbers[index2]
                            cos_angle = selected.angle_calculate(
                                index1, index2, index3)
                            distance = selected.length_calculate(
                                index2, index3)
                            radius_sum = periodic[
                                atomnum1].vdw_radius + periodic[atomnum2].vdw_radius
                            if cos_angle <= 0 and distance < radius_sum:
                                self.add_bond(
                                    index2, index3, "hydrogen", targeted)
                elif distance < 2.5 * radius_sum:
                    self.add_aux_bond(index_i, index_j, targeted)

    def _auto_angle_select(self, selected, targeted):
        for central_index in range(self.len):
            side_atoms = selected.bond[central_index]
            if len(side_atoms) >= 2:
                total_len = len(side_atoms)
                for left in range(total_len):
                    for right in range(left + 1, total_len):
                        self.add_angle(side_atoms[left], central_index, side_atoms[
                                       right], targeted)

    def _auto_dihed_select(self, selected, targeted):
        for cen_atom1 in range(self.len):
            connect_atoms1 = selected.bond[cen_atom1]
            if len(connect_atoms1) < 2:
                continue
            for cen_atom2 in connect_atoms1:
                connect_atoms2 = selected.bond[cen_atom2]
                if len(connect_atoms2) < 2:
                    continue
                maxatoms = 0  # initial max atoms connected to
                side_atom1 = -1
                for try_atom1 in connect_atoms1:
                    if try_atom1 == cen_atom2:
                        continue
                    allatoms = len(selected.bond[try_atom1])
                    if allatoms > maxatoms:
                        maxatoms = allatoms
                        side_atom1 = try_atom1
                for side_atom2 in connect_atoms2:
                    if side_atom2 == cen_atom1 or side_atom2 == side_atom1:
                        continue
                    self.add_dihed_new(side_atom1, cen_atom1,
                                       cen_atom2, side_atom2, targeted)

    def _hydrogen_halo_test(self, atomindex1, atomindex2):
        flag = False  # flag to indicate whether the two atoms can form a h-bond
        num1 = self.numbers[atomindex1]
        num2 = self.numbers[atomindex2]
        if num1 in TransitionSearch.halo_atom_numbers:
            if num2 == 1:
                index1 = atomindex1  # index 1 is halo, index 2 is H
                index2 = atomindex2
                flag = True
                return (flag, (index1, index2))
        elif num2 in TransitionSearch.halo_atom_numbers:
            if num1 == 1:
                index1 = atomindex2  # index 1 is halo, index 2 is H
                index2 = atomindex1
                flag = True
                return (flag, (index1, index2))
        return (flag,)

    def arrange_key_ic(self, *ic_index):
        for i in ic_index:
            self.ts_state.ic_switch(i, self._ic_key_counter)
            self._ic_key_counter += 1


class AtomsNumberError(Exception):
    pass


if __name__ == '__main__':
    fn_xyz = ht.context.get_fn("test/2h-azirine.xyz")
    mol = ht.IOData.from_file(fn_xyz)
    # print mol.numbers
    h22 = TransitionSearch(mol, mol)
    print(h22.numbers)
    h22.get_ts_guess_cc()
    # h22._auto_bond_select(h22.reagent, [h22.reagent])
    # print h22.reagent.aux_bond
    # h22._auto_angle_select(h22.reagent, [h22.reagent])
    h22.auto_ic_select(h22.reagent)
    print h22.reagent.bond
    print h22.reagent.ic_info
    print h22.reagent.procedures
    print h22.reagent.aux_bond
    h22.auto_ic_select(h22.reagent)
    print h22.reagent.bond
    # print h22.reagent.procedures
