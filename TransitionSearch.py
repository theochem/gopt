import numpy as np
import horton as ht

from copy import deepcopy
from horton.periodic import periodic
from saddle.ICTransformation import ICTransformation


class TransitionSearch(object):
    """Use to determine transition state structure and doing optimization for transition state to
    find a proper saddle point.

    Attributes:
        halo_atom_index (set): index of atoms that can form hydrogen bond
        halo_atom_numbers (tuple): atomic numbers that can form hydrogen bond
        len (int): numbers of atoms of system in total
        numbers (list): atomic numbers of each atoms in system with corresponding index number
        product (ICTransformation object): a ICTransformation object with product coordinates
        reactant (ICTransformation object): a ICTransformation object with reactant coordinates
        ts_state (ICTransformation object): a ICTransformation object with ts_state coordinates
    """

    def __init__(self, reactant, product):
        self.reactant = ICTransformation(reactant.coordinates)
        self.product = ICTransformation(product.coordinates)
        if reactant.numbers.tolist() != product.numbers.tolist():
            raise AtomsNumberError
        self.numbers = reactant.numbers
        self.len = len(self.numbers)
        self.halo_atom_index = set()
        for i in range(self.len):
            if self.numbers[i] in TransitionSearch.halo_atom_numbers:
                self.halo_atom_index.add(i)
        self.ts_state = None
        self._ic_key_counter = 0
        self._a_matrix = np.array([])
        self._b_perturb = np.array([])
        self._ts_dof = None 

    halo_atom_numbers = (7, 8, 9, 15, 16, 17)

    def _linear_check(self):
        ic_len = len(self.ts_state.ic)
        for i in range(ic_len):
            if self.ts_state.procedures[i][0] == "add_bend_angle":
                rad = self.ts_state.ic[i]
                if abs(np.sin(rad)) > 1e-3:
                    self._linear_struct_setting(False)
                    return
        self._linear_struct_setting(True)
        return

    def _linear_struct_setting(self, lnr_struct):
        if lnr_struct:
            self._ts_dof = 3 * self.len - 5
        else:
            self._ts_dof = 3 * self.len - 6

    def auto_ts_search(self, similar=None, ratio=0.5):
        """generate auto transition state initial geometry

        Args:
            similar (ICTransformation object, optional): select targeted geometry for geometry guess
            ratio (float, optional): ratio for combining reactant structure and product structure
        """
        if similar == None:
            similar = self.reactant
        self.auto_ic_select(similar, [self.reactant, self.product])
        self.ts_state = deepcopy(similar)
        self.ts_state.coordinates = self.get_ts_guess_cc(ratio)
        self.ts_state._reset_ic()
        self._linear_check()
        while len(self.ts_state.ic) < self._ts_dof:
            if self.ts_state.aux_bond:
                atom1, atom2 = self.ts_state.auto_upgrade_aux_bond()
                self.upgrade_aux_bond(atom1, atom2, [self.product, self.ts_state, self.reactant])
            else:
                print "something wrong"
                break

    def get_ts_guess_cc(self, ratio=0.5):
        """Summary

        Args:
            ratio (float, optional): ratio for combining reactant structure and product structure

        Raises:
            ValueError: value for ratio is beyond rational range
        """
        if ratio > 1. or ratio < 0.:
            raise ValueError
        ts_coordinate = self.reactant.coordinates * \
            ratio + self.product.coordinates * (1. - ratio)
        return ts_coordinate

    def get_ts_guess_ic(self, ratio=0.5):
        """Summary

        Args:
            ratio (float, optional): ratio for combining reactant structure and product structure

        Raises:
            AtomsNumberError: reactant and product sample geometry have different number
                of internal coordinates
            ValueError: value for ratio is beyond rational range
        """
        if len(self.reactant.ic) != len(self.product.ic):
            raise AtomsNumberError
        if ratio > 1. or ratio < 0.:
            raise ValueError
        target_ic = self.reactant.ic * ratio + self.product.ic * (1. - ratio)
        self.ts_state.target_ic = target_ic

    @staticmethod
    def add_bond(atom1, atom2, b_type, multistructure):
        """add bond to several geometry structure

        Args:
            atom1 (int): index number of atom1
            atom2 (int): index number of atom2
            b_type (string): chemical bond type
            multistructure (tuple): a tuple of structure
        """
        for structure in multistructure:
            structure.add_bond_length(atom1, atom2, b_type)

    @staticmethod
    def add_angle(atom1, atom2, atom3, multistructure):
        """add angle to several geometry structure

        Args:
            atom1 (int): index number of atom1
            atom2 (int): index number of atom2
            atom3 (int): index number of atom3
            multistructure (tuple): a tuple of structure
        """
        for structure in multistructure:
            structure.add_bend_angle(atom1, atom2, atom3)

    @staticmethod
    def add_dihed_conv(atom1, atom2, atom3, atom4, multistructure):
        """add conventional dihedral angle to several geometry structure

        Args:
            atom1 (int): index number of atom1
            atom2 (int): index number of atom2
            atom3 (int): index number of atom3
            atom4 (int): index number of atom4
            multistructure (tuple): a tuple of structure
        """
        for structure in multistructure:
            structure.add_dihed_angle(atom1, atom2, atom3, atom4)

    @staticmethod
    def add_dihed_new(atom1, atom2, atom3, atom4, multistructure):
        """add new dihedral descriptors to several geometry structure

        Args:
            atom1 (int): index number of atom1
            atom2 (int): index number of atom2
            atom3 (int): index number of atom3
            atom4 (int): index number of atom4
            multistructure (tuple): a tuple of structure
        """
        for structure in multistructure:
            structure.add_dihed_new(atom1, atom2, atom3, atom4)

    @staticmethod
    def add_aux_bond(atom1, atom2, multistructure):
        """add aux bond in several geometry structure

        Args:
            atom1 (int): index number of atom1
            atom2 (int): index number of atom2
            multistructure (tuple): a tuple of structure
        """
        for structure in multistructure:
            structure.add_aux_bond(atom1, atom2)

    @staticmethod
    def upgrade_aux_bond(atom1, atom2, multistructure):
        """upgrade an aux bond to several geometry structure

        Args:
            atom1 (int): index number of atom1
            atom2 (int): index number of atom2
            multistructure (tuple): a tuple of structure
        """
        for structure in multistructure:
            structure.upgrade_aux_bond(atom1, atom2)

    def auto_ic_select(self, selected_structure, target_structure=[]):
        """select internal coordinates according to the selected_structure

        Args:
            selected_structure (ICTransformation object): target model structure
            target_structure (list, optional): list of structure to add internal coordinates
        """
        assert isinstance(target_structure,
                          list), "target_structure should be a list"
        if not target_structure:
            target_structure.append(selected_structure)
        self._auto_bond_select(selected_structure, target_structure)
        self._auto_angle_select(selected_structure, target_structure)
        self._auto_dihed_select(selected_structure, target_structure)

    def auto_ic_select_combine(self):
        self.auto_ic_select(self.reactant, [self.reactant, self.product])
        self.auto_ic_select(self.product, [self.reactant, self.product])

    def _auto_bond_select(self, selected, targeted):
        """Auto bond generator

        Args:
            selected (ICTransformation object): model structure for add internal coordinates
            targeted (ICTransformation object): target structure to add internal coordinates
        """
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
        """auto angle generator

        Args:
            selected (ICTransformation object): model structure for add internal coordinates
            targeted (ICTransformation object): target structure to add internal coordinates
        """
        for central_index in range(self.len):
            side_atoms = selected.bond[central_index]
            if len(side_atoms) >= 2:
                total_len = len(side_atoms)
                for left in range(total_len):
                    for right in range(left + 1, total_len):
                        self.add_angle(side_atoms[left], central_index, side_atoms[
                                       right], targeted)

    def _auto_dihed_select(self, selected, targeted):
        """auto dihed generator

        Args:
            selected (ICTransformation object): model structure for add internal coordinates
            targeted (ICTransformation object): target structure to add internal coordinates
        """
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
        """check whether bond between atomindex and atomindex2 can form a hydrogen bond later,
        if so, return the flag and index of two atoms, nonhydrogen atom index first, then the index of hydrogen
        if not, only return the flag
        Args:
            atomindex1 (int): index of atom1
            atomindex2 (TYPE): index of atom2

        Returns:
            tuple: (flag [,tuple(index1, index2)])

        """
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

    def auto_key_ic_select(self):
        """auto key internal coordinates generator
        """
        key_ic = []
        for i in range(len(self.ts_state.ic)):  # i is the index of ic of ts_state
            procedure = self.ts_state.procedures[i]
            if procedure[0] == "add_bond_length":
                atomindex1, atomindex2 = procedure[1]
                atomnum1 = self.numbers[atomindex1]
                atomnum2 = self.numbers[atomindex2]
                threshhold = (periodic[atomnum1].cov_radius +
                              periodic[atomnum2].cov_radius) * 0.5
                if (abs(self.reactant.ic[i] - self.product.ic[i]) > threshhold or
                        abs(self.reactant.ic[i] - self.ts_state.ic[i]) > threshhold or
                        abs(self.product.ic[i] - self.ts_state.ic[i]) > threshhold):
                    key_ic.append(i)
            if procedure[0] == "add_bend_angle":
                atomindex1, atomindex2, atomindex3 = procedure[1]
                atomnum1 = self.numbers[atomindex1]
                atomnum2 = self.numbers[atomindex2]
                atomnum3 = self.numbers[atomindex3]
                threshhold = 0.5236  # 1rad = 57.2958 degrees, therefore 30degree = 0.5236rad
                if (abs(self.reactant.ic[i] - self.product.ic[i]) > threshhold or
                        abs(self.reactant.ic[i] - self.ts_state.ic[i]) > threshhold or
                        abs(self.product.ic[i] - self.ts_state.ic[i]) > threshhold):
                    key_ic.append(i)
        print "key ic",key_ic
        self._arrange_key_ic(key_ic)

    def _arrange_key_ic(self, ic_index):
        """rearrange the sequence of internal key internal coordinates

        Args:
            ic_index (int): index of internal coordinates
        """
        for i in ic_index:
            self.ts_state.ic_swap(i, self._ic_key_counter)
            self._ic_key_counter += 1

    # def _matrix_a_eigen(self):
    #     """calculate eigenvalue of b_matrix, select 3n-5 to form the a matrix

    #     Returns:
    #         numpy.array: shape(3N - 5, n), A matrix
    #     """
    #     matrix_space = np.dot(self.ts_state.b_matrix, self.ts_state.b_matrix.transpose())
    #     eig_value, eig_vector = np.linalg.eig(matrix_space)
    #     ic_len = len(self.ts_state.ic)
    #     a_matrix = np.zeros((self._ts_dof, ic_len), float)
    #     counter = 0
    #     for i in len(eig_value):
    #         if eig_value[i] < 0.01:
    #             continue
    #         a_matrix[counter] = eig_value[:, i]
    #         counter += 1
    #         if counter >= (self._ts_dof):
    #             break
    #     return a_matrix

    # def _projection(self):
    #     """project perturbation on each key internal coordinates into relizable internal coordinates

    #     Returns:
    #         numpy.array: shape(n, R)
    #     """
    #     b_matrix = self.ts_state.b_matrix
    #     b_pinv = np.linalg.pinv(b_matrix)
    #     prj_matrix = np.dot(b_matrix, b_pinv)
    #     ic_len = len(self.ts_state.ic)
    #     ic_keyic_len = self._ic_key_counter
    #     e_perturb = np.identity(ic_keyic_len)
    #     b_perturb = np.dot(prj_matrix, e_perturb)
    #     return b_perturb

    # @staticmethod
    # def _gram_ortho(vectors, transpose=False):
    #     """grammian orthogonal treatment, to orthogonize the row space
        
    #     Args:
    #         vectors (numpy.array): a set of vectors to be orthogonized
    #         transpose (bool, optional): if the vactor span a column space, true
    #             to transpose it into row space
        
    #     Returns:
    #         numpy.array: orthogonlized vectors set. span in row space.
    #     """
    #     if transpose:
    #         vectors = vectors.T
    #     vec_len = len(vectors)
    #     gram = np.zeros((vec_len, vec_len), float)
    #     for row in range(vec_len):
    #         for column in range(vec_len):
    #             gram[row][column] = np.dot(vectors[row], vectors[column])
    #     eig_value, eig_vector = np.linalg.eig(gram)
    #     basisset = np.zeros((vec_len, vec_len), float)
    #     counter = 0
    #     for i in range(vec_len):
    #         if eig_value[i] > 0.01:
    #             basisset[counter] = eig_value[:, i]
    #             counter += 1
    #     return basisset[:counter]

    # def _deloc_reduce_ic(self):
    #     """orthogonize perturbation, calculate reduced internal coordinates for key ic
        
    #     Returns:
    #         numpy.array: reduced internal coordinates
    #     """
    #     b_perturb = self._projection()
    #     basisset = self._gram_ortho(b_perturb)
    #     reduced_ic = np.dot(b_perturb, basisset)
    #     return reduced_ic

    # def _deloc_non_reduce_ic(self):
    #     """calculate nonreduced_space by project a_matrix to nonspace of reduced space
        
    #     Returns:
    #         numpy.array: nonreduced vectors to form nonreduced space
    #     """
    #     a_matrix = self._matrix_a_eigen()
    #     v_reduce = self._deloc_reduce_ic()
    #     reduced_space_1 = np.dot(v_reduce, v_reduce.T)
    #     reduced_space_2 = np.dot(non_reduced_space_1, a_matrix.T)
    #     nonreduced_space = a_matrix - reduced_space_2
    #     return nonreduced_space

    # def _nonreduce_ic(self):
    #     """calculate nonreduce internal coordinates
        
    #     Returns:
    #         numpy.array: nonreduced internal coordinates
    #     """
    #     d_vectors = self._deloc_non_reduce_ic()
    #     basisset = self._gram_ortho(d_vectors)
    #     nonreduce_ic = np.dot(d_vectors, basisset)
    #     return nonreduce_ic

    # def get_v_basis(self):
    #     """get 3n-5 nonredundant internal coordinates
        
    #     Returns:
    #         numpy.array: nonredundant internal coordinates
    #     """
    #     reduced = self._deloc_reduce_ic()
    #     non_reduced = self._nonreduce_ic()
    #     return np.vstack((reduced, non_reduced))


class AtomsNumberError(Exception):
    pass


if __name__ == '__main__':
    fn_xyz = ht.context.get_fn("test/Br_HCl.xyz")
    fn_xyz_2 = ht.context.get_fn("test/Cl_HBr.xyz")
    reactant = ht.IOData.from_file(fn_xyz)
    product = ht.IOData.from_file(fn_xyz_2)
    # print mol.numbers
    h22 = TransitionSearch(reactant, product)
    print(h22.numbers)
    h22.auto_ic_select_combine()
    h22.auto_ts_search()
    h22.auto_key_ic_select()
    print "ic",h22.ts_state.ic
    print "ic_reactant", h22.reactant.ic
    print "ic_prodect", h22.product.ic
    print "bond",h22.ts_state.bond
    print "ic info",h22.ts_state.ic_info
    print "proce",h22.ts_state.procedures
    print "aux",h22.ts_state.aux_bond
    print h22._ts_dof
    print "key ic number", h22._ic_key_counter
