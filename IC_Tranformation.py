from IC_Functions import *
from Cost_Functions import *
import numpy as np
import horton as ht
import copy


class IC_Transformation(object):
    """IC_Transformation is a class for coordinates transformation from cartesian coordinates to internal 
    coordinates and vise versa.

    Arguments:
     | ``molecule`` is a object of Horton IODate class

    Attributes:
     | ``coordinates`` -- cartesian coordinates of molecule in N*3 numpy array.
     | ``numbers`` -- a list of atomic number of each atoms of target molecule.
     | ``ic`` -- a list of internal coordinates calculated through transformation.
     | ``iteration_falg`` -- a Bool flag to mark is it doing iteration or generate new ic. T for iteration, F for new ic.
     | ``procedures`` -- a list of steps used to generate ic which will be used when doing iteration
     | ``bond`` -- a list of added ic bond set
     | ``angle`` -- a list of added ic angle set
     | ``dihed`` -- a list of added ic dihed set 

    """

    def __init__(self, molecule):
        self.coordinates = molecule.coordinates
        self.numbers = molecule.numbers
        self.len = len(self.numbers)
        self.ic = []
        self.iteration_flag = False
        self.procedures = []
        self.bond = []
        self.angle = []
        self.dihed = []
        self.B_matrix = np.zeros((0,3*self.len),float)
        self._ic_differences = []
        self._target_ic = []
        self.H_matrix = np.zeros((0, 3*self.len, 3*self.len), float)


    def add_bond_length(self, atom1, atom2, b_type = "covalence"):
        """To add a bond between atom1 and atom2

        Arguments:
         | ``atom1`` the index of atom1 in self.numbers
         | ``atom2`` the index of atom2 in self.numbers
        """
        atoms = (atom1, atom2)
        info = "add_bond_length"
        if self._repetition_check(atoms, b_type):
            self._add_ic(info, atoms)


    def add_bend_angle(self, atom1, atom2, atom3):
        """To add an angle between atom1, atom2 and atom3.
        1-2-3, atom2 is the central atom of the angle

        Arguments:
         | ``atom1`` the index of atom1 in self.numbers
         | ``atom2`` the index of atom2 in self.numbers
         | ``atom3`` the index of atom3 in self.numbers
        """        
        atoms = (atom1, atom2, atom3)
        info = "add_bend_angle"
        if self._repetition_check(atoms):
            self._add_ic(info, atoms)


    def add_dihed_angle(self, atom1, atom2, atom3, atom4):
        """To add an conventional angle between atom1, atom2, atom3 and atom4.
           1-2-3-4, atom 2 and 3 is the central atoms 

        Arguments:
         | ``atom1`` the index of atom1 in self.numbers
         | ``atom2`` the index of atom2 in self.numbers
         | ``atom3`` the index of atom3 in self.numbers
         | ``atom4`` the index of atom4 in self.numbers
        """ 
        atoms = (atom1, atom2, atom3, atom4)
        info = "add_dihed_angle"
        d_type = "conventional"      
        if self._repetition_check(atoms, d_type):
            self._add_ic(info, atoms)


    def add_dihed_new(self, atom1, atom2, atom3, atom4):
        """To add an new dihedral angle indicator between atom1, atom2, atom3 and atom4.
           1-2-3-4, atom 2 and 3 is the central atoms 

        Arguments:
         | ``atom1`` the index of atom1 in self.numbers
         | ``atom2`` the index of atom2 in self.numbers
         | ``atom3`` the index of atom3 in self.numbers
         | ``atom4`` the index of atom4 in self.numbers
        """ 
        atoms = (atom1, atom2, atom3, atom4)
        d_type = "new"
        if self._repetition_check(atoms, d_type):
            self._add_dihed_angle_new_dot(atoms)
            self._add_dihed_angle_new_cross(atoms)


    def _add_dihed_angle_new_dot(self, atoms):
        """private method to add an new dihedral angle dot indicator between atom1, atom2, atom3 and atom4.
           1-2-3-4, atom 2 and 3 is the central atoms 
        """ 
        info = "add_dihed_angle_new_dot"
        self._add_ic(info, atoms)


    def _add_dihed_angle_new_cross(self, atoms):
        """private method to add an new dihedral angle dot indicator between atom1, atom2, atom3 and atom4.
           1-2-3-4, atom 2 and 3 is the central atoms 
        """ 
        info = "add_dihed_angle_new_cross"
        self._add_ic(info, atoms)


    def _repetition_check(self, atoms, type = "default"):
        """private method to check whether the newly add ic has already existed or not
        """
        atoms_len = len(atoms)
        content = ()
        if atoms_len == 2:
            content = (set(atoms), type)
            if content not in self.bond:
                self.bond.append(content)
                return True
            else: return False

        if atoms_len == 3:
            content = (atoms[1], set([atoms[0], atoms[2]]), type)
            if content not in self.angle:
                self.angle.append(content)
                return True
            else: return False
    
        if atoms_len == 4:
            content = (set([atoms[1], atoms[2]]), set([atoms[0], atoms[3]]), type)
            if content not in self.dihed:
                self.dihed.append(content)
                return True
            else: return False 


    def _add_ic(self, info, atoms):
        """ ic was added through this private method.
        """
        procedures = (info, atoms)
        ic_function = IC_Transformation._IC_types[info]
        coordinates = self.get_coordinates(atoms)
        result, d, dd = ic_function(coordinates, deriv = 2)
        if not self.iteration_flag:
            self.procedures.append(procedures)
        self.ic.append(result)
        self._add_B_matrix(d, atoms)
        self._add_H_matrix(dd, atoms)


    def _add_B_matrix(self, deriv, atoms):
        """calculated B matrix for object
        """
        tmp_B_matrix = np.zeros((len(self.ic), 3*self.len), float)
        tmp_B_matrix[:-1, :] = self.B_matrix
        for i in range(len(atoms)):
            tmp_B_matrix[-1, 3*atoms[i]: 3*atoms[i]+3] += deriv[i]
        self.B_matrix = np.zeros((len(self.ic), 3*self.len), float)
        self.B_matrix[:,:] = tmp_B_matrix


    def _add_H_matrix(self, deriv2, atoms):
        tmp_H_matrix = np.zeros((len(self.ic), 3*self.len, 3*self.len), float)
        tmp_H_matrix[:-1, : , : ] = self.H_matrix
        for i in range(len(atoms)):
            for j in range(3):
                tmp_H_matrix[-1, 3*atoms[i]+j, 3*atoms[i]: 3*atoms[i]+3] += deriv2[i][j][i]
        self.H_matrix = np.zeros((len(self.ic), 3*self.len, 3*self.len), float)
        self.H_matrix = tmp_H_matrix


    def get_coordinates(self, atoms):
        """private method to retrive atoms' cartesian coordinates
        """
        atom_length = len(atoms)
        coordinates = np.array([])
        for i in atoms:
            coordinates = np.append(coordinates, self.coordinates[i])
        return coordinates.reshape(-1, 3)


    def _get_difference(self):
        self._ic_differences.extend([0]*(len(self.ic) - len(self._ic_differences)))
        return list(np.array(self.ic) - np.array(self.target_ic))


    def _set_difference(self, value):
        self.target_ic = list(np.array(self.ic) + np.array(value))


    ic_differences = property(_get_difference, _set_difference)


    def _get_target_ic(self):
        if len(self._target_ic) < len(self.ic):
            self._target_ic.extend(self.ic[len(self._target_ic):])
        return self._target_ic or self.ic


    def _set_target_ic(self, value):
        if len(value) == len(self.ic):
            self._target_ic = value


    target_ic = property(_get_target_ic, _set_target_ic)


    @property
    def weight_matrix(self):
        _weight_matrix = np.zeros((len(self.ic), len(self.ic)), float)
        for i in range(len(self.procedures)):
            info = self.procedures[i][0]
            atoms = self.procedures[i][1]
            if info == "add_dihed_angle":
                ic_function = IC_Transformation._IC_types["add_bend_angle"]
                coordinates1 = self.get_coordinates((atoms[0], atoms[1], atoms[2]))
                angle1 = ic_function(coordinates1, deriv = 0)[0]
                coordinates2 = self.get_coordinates((atoms[1], atoms[2], atoms[3]))
                angle2 = ic_function(coordinates2, deriv = 0)[0]
                _weight_matrix[i][i] = np.sin(angle1)**2 * np.sin(angle2)**2
            else:
                _weight_matrix[i][i] = 1
        return _weight_matrix


    def calculate_cost(self):
        c_value = 0
        ic_num = len(self.ic)
        c_deriv = np.zeros(ic_num, float)
        c_deriv_2 = np.zeros((ic_num, ic_num), float) 
        for i in range(ic_num):
            func = IC_Transformation._IC_costs_value[self.procedures[i][0]]
            deriv_func = IC_Transformation._IC_costs_diff[self.procedures[i][0]]
            deriv_2_func = IC_Transformation._IC_costs_diff_2[self.procedures[i][0]]
            origin = self.ic[i]
            target = self.target_ic[i]
            c_value += func(origin, target) * self.weight_matrix[i][i]
            c_deriv[i] += deriv_func(origin, target) * self.weight_matrix[i][i]
            c_deriv_2[i][i] += deriv_2_func(origin, target) * self.weight_matrix[i][i]
        c_x_deriv = np.dot(c_deriv, self.B_matrix)
        c_x_deriv_2_part_1 = np.dot(np.dot(np.transpose(self.B_matrix), c_deriv_2), self.B_matrix)
        c_x_deriv_2_part_2 = np.tensordot(c_deriv, self.H_matrix, 1)
        c_x_deriv_2 = c_x_deriv_2_part_1 + c_x_deriv_2_part_2
        return c_value, c_x_deriv, c_x_deriv_2






    _IC_types = {
        "add_bond_length":IC_Functions.bond_length,
        "add_bend_angle":IC_Functions.bend_angle,
        "add_dihed_angle":IC_Functions.dihed_angle,
        "add_dihed_angle_new_dot":IC_Functions.dihed_angle_new_dot,
        "add_dihed_angle_new_cross":IC_Functions.dihed_angle_new_cross
    }
        
    
    _IC_costs_diff = {
        "add_bond_length":Cost_Functions.direct_diff,
        "add_bend_angle":Cost_Functions.cos_diff,
        "add_dihed_angle":Cost_Functions.dihed_diff,
        "add_dihed_angle_new_dot":Cost_Functions.direct_diff,
        "add_dihed_angle_new_cross":Cost_Functions.direct_diff
    }
    

    _IC_costs_value = {
        "add_bond_length":Cost_Functions.direct_square,
        "add_bend_angle":Cost_Functions.cos_square,
        "add_dihed_angle":Cost_Functions.dihed_square,
        "add_dihed_angle_new_dot":Cost_Functions.direct_square,
        "add_dihed_angle_new_cross":Cost_Functions.direct_square
    }

    _IC_costs_diff_2 = {
        "add_bond_length": Cost_Functions.direct_diff_2,
        "add_bend_angle": Cost_Functions.cos_diff_2,
        "add_dihed_angle": Cost_Functions.dihed_diff_2,
        "add_dihed_angle_new_dot": Cost_Functions.direct_diff_2,
        "add_dihed_angle_new_cross": Cost_Functions.direct_diff_2
    }



if __name__ == '__main__':
    fn_xyz = ht.context.get_fn('test/2h-azirine.xyz')
    mol = ht.IOData.from_file(fn_xyz)
    cc_object = IC_Transformation(mol)
    print cc_object.coordinates
    print cc_object.numbers
    cc_object.add_bond_length(0,1) ## 1st ic
    print cc_object.ic
    print cc_object.procedures[0][1]
    cc_object.add_bond_length(0,1)
    print cc_object.ic
    cc_object.add_bend_angle(1,2,3) ## 2nd ic
    print cc_object.ic
    print cc_object.procedures
    cc_object.add_dihed_angle(1,2,3,4) ## 3rd ic
    print cc_object.ic
    cc_object.add_dihed_new(4,3,2,1) ## 4th, 5th ic
    # cc_object.add_dihed_new(1,2,3,4)
    print cc_object.ic
    print cc_object.bond
    # cc_object.ic = []
    # cc_object.B_matrix = B_matrix = np.zeros((0,3*cc_object.len),float)
    # print cc_object.ic
    # cc_object.iteration_flag = True
    print cc_object.procedures
    # for i in cc_object.procedures:
    #     cc_object._add_ic(i[0], i[1])
    #     print i
    print cc_object.ic
    print cc_object.ic_differences
    cc_object.ic_differences = [0.2,0.2,0.1,0.0,0.0]
    print cc_object.target_ic
    print cc_object.ic_differences
    # cc_object.add_bond_length(1,2) ## 6th ic
    print cc_object.ic
    print cc_object.ic_differences
    # cc_object.add_bond_length(2,3)
    print cc_object.ic
    print cc_object.B_matrix
    a,b,c = cc_object.calculate_cost()
    print c.shape
    print cc_object.H_matrix.shape
