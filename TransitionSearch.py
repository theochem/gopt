import numpy as np
import horton as ht
from horton.periodic import periodic
from saddle.IC_Transformation import IC_Transformation


class TransitionSearch(object):
    def __init__(self, reagent, product):
        self.reagent = IC_Transformation(reagent.coordinates)
        self.product = IC_Transformation(product.coordinates)
        if reagent.numbers.tolist() != product.numbers.tolist():
            raise AtomsNumberError
        self.numbers = reagent.numbers
        self.len = len(self.numbers)
        self.h_atom_index = set()
        self.halo_atom_index = set()
        for i in range(self.len):
            if self.numbers[i] in TransitionSearch.hydro_bond_numbers:
                self.halo_atom_index.add(i)


    hydro_bond_numbers = (7, 8, 9, 15, 16, 17)


    def add_bond(self, atom1, atom2, b_type):
        self.reagent.add_bond_length(atom1, atom2, b_type)
        self.product.add_bond_length(atom1, atom2, b_type)


    def add_angle(self, atom1, atom2, atom3):
        self.reagent.add_bend_angle(atom1, atom2, atom3)
        self.product.add_bend_angle(atom1, atom2, atom3)


    def add_dihed_conv(self, atom1, atom2, atom3, atom4):
        self.reagent.add_dihed_angle(atom1, atom2, atom3, atom4)
        self.product.add_dihed_angle(atom1, atom2, atom3, atom4)


    def add_dihed_new(self, atom1, atom2, atom3, atom4):
        self.reagent.add_dihed_new(atom1, atom2, atom3, atom4)
        self.product.add_dihed_new(atom1, atom2, atom3, atom4)


    def add_aux_bond(self, atom1, atom2):
        self.reagent.add_aux_bond(atom1, atom2)
        self.product.add_aux_bond(atom1, atom2)


    def upgrade_aux_bond(self, atom1, atom2):
        self.reagent.upgrade_aux_bond(atom1, atom2)
        self.product.upgrade_aux_bond(atom1, atom2)


    def auto_ic_select(self):
        for index_i in range(self.len): #i,j is index of selected atoms in coordinates
            for index_j in range(i+1, self.len): 
                atomnum1 = self.numbers[index_i] #atomnum1, atomnum2 is atomic number of two selected atom2
                atomnum2 = self.numbers[index_j]
                length = self.reagent.length_calculate(index_i, index_j)
                radius_sum = periodic[atomnum1].cov_radius + periodic[atomnum2].cov_radius
                if length < 1.3 * radius_sum:
                    self.reagent.add_bond_length(index_i, index_j, "covalence")
                    if atomnum1 in hydro_bond_numbers:
                        self.halo_atom_index.add(atomnum1)
                        if atomnum2 == 1:
                            self.h_atom_index.add(atomnum2)
                    elif atomnum2 in hydro_bond_numbers:
                        self.halo_atom_index.add(atomnum2)
                        if atomnum1 == 1:
                            self.h_atom_index.add(atomnum1)


    def hydrogen_halo_test(self, atomindex1, atomindex2):
        flag = False #flag to indicate whether the two atoms can form a h-bond
        num1 = self.numbers[atomindex1]
        num2 = self.numbers[atomindex2]
        if num1 in hydro_bond_numbers:
            if num2 == 1:
                flag = True
        elif num2 in hydro_bond_numbers:
            if num1 == 1:
                tem = num2 #switch the value of num1 and num2, num1 is Halo, num2 is H
                num2 = num1
                num1 = tem
                flag = True
        if flag:
            for index3 in self.halo_atom_index:
                pass



    # auto_select = False

    # @staticmethod
    # def length_compair(coor, length, judge_func, execute_func):
    #     for i in range(0, length-1):
    #         for j in range(i+1, length):
    #             if judge_func(i,j):
    #                 execute_func(i, j)





class AtomsNumberError(Exception):
    pass



if __name__ == '__main__':
    fn_xyz = ht.context.get_fn("test/2h-azirine.xyz")
    mol = ht.IOData.from_file(fn_xyz)
    # print mol.numbers
    h22 = TransitionSearch(mol,mol)
    print h22.numbers
