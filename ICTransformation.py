import numpy as np
import saddle.optimizer as op

from saddle.ICFunctions import ICFunctions
from saddle.CostFunctions import CostFunctions


__all__ = ["ICTransformation"]


class ICTransformation(object):
    """ICTransformation is a class for coordinates transformation from cartesian coordinates to internal
    coordinates and vise versa.

    Arguments:
        coordinates (numpy.array): shape(N, 3), a numpy array which contains cartesian coordinates of atoms, shape is (m, 3n)

    Attributes:
        angle (list): a list of added ic angle
        aux_bond (list): a list of added auxiliary bond.
        B_matrix (numpy.array): shape(m, 3N), first derivative for coordinates transformation from cartesian to internal
        bond (list): list of added ic bond
        coordinates (numpy.array): shape(N, 3), cartesian coordinates of molecule
        dihed (list): list of added dihedral
        H_matrix (numpy.array): shape(m, 3N, 3N), second derivative for coordinates transformation from cartesian to internal
        ic (numpy.array): shape(m,), internal coordinates calculated throught transformation
        ic_differences (numpy.array): shape(m,), difference between actual internal coordinates and target internal coordinates
        iteration_flag (bool): a flag to mart is it doing iteration or calculate new ic
        len (int): number of atoms
        procedures (list): steps used to generate ic which will be used when doing iteration
        target_ic (numpy.array): shape(m,), target ic
    """

    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.len = len(coordinates.reshape(-1, 3))
        self.ic = np.array([])
        self.ic_info = []
        self.iteration_flag = False
        self.procedures = []
        self.bond =[[] for i in range(self.len)]
        self._re_bond = []
        self._angle = []
        self._dihed = []
        self.b_matrix = np.zeros((0, 3 * self.len), float)
        self._ic_differences = np.array([])
        self._target_ic = np.array([])
        self.h_matrix = np.zeros((0, 3 * self.len, 3 * self.len), float)
        self.aux_bond = []

    def length_calculate(self, atom1, atom2):
        """To calculate distance between two atoms

        Args:
            atom1 (int): index of atom1 in coordinates
            atom2 (int): index of atom1 in coordinates

        Returns:
            float: distance between two atoms
        """
        dis_array = self.coordinates[atom1] - self.coordinates[atom2]
        return np.linalg.norm(dis_array)

    def angle_calculate(self, atom1, atom2, atom3):
        """To calculate cosine value of the angle consist of atom1, atom2, atom3

        Args:
            atom1 (int): index of atom1 in coordinates
            atom2 (int): index of atom2 in coordinates
            atom3 (int): index of atom3 in coordinates

        Returns:
            float: the cosine value of angle
        """
        array1 = self.coordinates[atom1] - self.coordinates[atom2]
        array2 = self.coordinates[atom3] - self.coordinates[atom2]
        cos_angle = np.dot(array1, array2) / \
            (np.linalg.norm(array1) * np.linalg.norm(array2))
        return cos_angle

    def add_bond_length(self, atom1, atom2, b_type="covalence"):
        """To add a bond between atom1 and atom2

        Arguments:
            atom1 (int): index of atom1 in coordinates
            atom2 (int): index of atom2 in coordinates
            b_type (str, optional): bond type infomation.
        """
        atoms = (atom1, atom2)
        bond_type = b_type
        info = "add_bond_length"
        if self._repetition_check(atoms):
            self.ic_info.append(bond_type)
            self._add_ic(info, atoms)
            self.bond[atom1].append(atom2)
            self.bond[atom2].append(atom1)

    def add_aux_bond(self, atom1, atom2):
        """To add a auxiliary bond to self.aux_bond reservoir

        Arguments:
            atom1 (int): index of atom1 in coordinates
            atom2 (int): index of atom2 in coordinates
        """
        atoms = (atom1, atom2)
        if self._repetition_check(atoms):
            self.aux_bond.append(atoms)

    def upgrade_aux_bond(self, atom1, atom2, b_type="auxiliary"):
        """To upgrade an auxiliary bond to ic

        Arguments:
            atom1 (int): index of atom1
            atom2 (int): index of atom2
            b_type (str, optional): bond type information
        """
        atoms = (atom1, atom2)
        bond_type = b_type
        info = "add_bond_length"
        if atoms in self.aux_bond:
            self.ic_info.append(bond_type)
            self._add_ic(info, atoms)

    def add_bend_angle(self, atom1, atom2, atom3):
        """To add an angle between atom1, atom2 and atom3.
        1-2-3, atom2 is the central atom of the angle

        Arguments:
            atom1 (int): index of atom1
            atom2 (int): index of atom2
            atom3 (int): index of atom3
        """
        atoms = (atom1, atom2, atom3)
        angle_type = "normal angle"
        info = "add_bend_angle"
        if self._repetition_check(atoms):
            self.ic_info.append(angle_type)
            self._add_ic(info, atoms)

    def add_dihed_angle(self, atom1, atom2, atom3, atom4):
        """To add an conventional angle between atom1, atom2, atom3 and atom4.
           1-2-3-4, atom 2 and 3 is the central atoms

        Arguments:
            atom1 (int): index of atom1
            atom2 (int): index of atom2
            atom3 (int): index of atom3
            atom4 (int): index of atom4
        """
        atoms = (atom1, atom2, atom3, atom4)
        info = "add_dihed_angle"
        d_type = "conventional dihedral"
        if self._repetition_check(atoms, d_type):
            self.ic_info.append(d_type)
            self._add_ic(info, atoms)

    def add_dihed_new(self, atom1, atom2, atom3, atom4):
        """To add an new dihedral angle indicator between atom1, atom2, atom3 and atom4.
           1-2-3-4, atom 2 and 3 is the central atoms

        Arguments:
            atom1 (int): index of atom1
            atom2 (int): index of atom2
            atom3 (int): index of atom3
            atom4 (int): index of atom4
        """
        atoms = (atom1, atom2, atom3, atom4)
        d_type = "new dihedral"
        if self._repetition_check(atoms, d_type):
            self._add_dihed_angle_new_dot(atoms)
            self._add_dihed_angle_new_cross(atoms)

    def _add_dihed_angle_new_dot(self, atoms):
        """private method to add an new dihedral angle dot indicator between atom1, atom2, atom3 and atom4.
           1-2-3-4, atom 2 and 3 is the central atoms

        Arguments:
            atoms (tuple): a tuple of atoms indexes
        """

        info = "add_dihed_angle_new_dot"
        self.ic_info.append("new dihedral doc")
        self._add_ic(info, atoms)

    def _add_dihed_angle_new_cross(self, atoms):
        """private method to add an new dihedral angle dot indicator between atom1, atom2, atom3 and atom4.
           1-2-3-4, atom 2 and 3 is the central atoms

        Arguments:
            atoms (tuple): a tuple of atoms indexes
        """

        info = "add_dihed_angle_new_cross"
        self.ic_info.append("new dihedral cross")
        self._add_ic(info, atoms)

    def _repetition_check(self, atoms, ictype="default"):
        """private method to check whether the newly add ic has already existed or not

        Arguments:
            atoms (tuple): a tuple of atoms indexes
            type (str, optional): additional ic coordinates for check repetition

        Returns: bool, false if it is existed, otherwise true
        """
        atoms_len = len(atoms)
        content = ()
        if atoms_len == 2:
            content = (set(atoms))
            if content not in self._re_bond:
                self._re_bond.append(content)
                return True
            else:
                return False
        if atoms_len == 3:
            content = (atoms[1], set([atoms[0], atoms[2]]), ictype)
            if content not in self._angle:
                self._angle.append(content)
                return True
            else:
                return False
        if atoms_len == 4:
            content = (set([atoms[1], atoms[2]]),
                       set([atoms[0], atoms[3]]), ictype)
            if content not in self._dihed:
                self._dihed.append(content)
                return True
            else:
                return False

    def _add_ic(self, info, atoms):
        """ic was added through this private method.

        Arguments:
            info (str): ic coordinates information
            atoms (tuple): a tuple of atoms indexes
        """
        procedures = (info, atoms)
        ic_function = ICTransformation._IC_types[info]
        coordinates = self.get_coordinates(atoms)
        result, d, dd = ic_function(coordinates, deriv=2)
        if not self.iteration_flag:
            self.procedures.append(procedures)
        self.ic = np.append(self.ic, result)
        self._add_b_matrix(d, atoms)
        self._add_h_matrix(dd, atoms)

    def _add_b_matrix(self, deriv, atoms):
        """calculated B matrix for object

        Arguments:
            deriv (numpy.array): first derivative of each atoms when doing coordinates transformation
            atoms (tuple): a tuple of atoms indexes
        """
        tmp_b_matrix = np.zeros((len(self.ic), 3 * self.len), float)
        tmp_b_matrix[:-1, :] = self.b_matrix
        for i in range(len(atoms)):
            tmp_b_matrix[-1, 3 * atoms[i]: 3 * atoms[i] + 3] += deriv[i]
        self.b_matrix = np.zeros((len(self.ic), 3 * self.len), float)
        self.b_matrix[:, :] = tmp_b_matrix

    def _add_h_matrix(self, deriv2, atoms):
        """calculate H matrix for coordinates transformation from cartesian to internal

        Arguments:
            deriv2 (numpy.array): second derivative of each atoms when doing transformation
            atoms (tuple): a tuple of atoms indexes
        """
        tmp_h_matrix = np.zeros(
            (len(self.ic), 3 * self.len, 3 * self.len), float)
        tmp_h_matrix[:-1, :, :] = self.h_matrix
        for i in range(len(atoms)):
            for j in range(3):
                tmp_h_matrix[-1, 3 * atoms[i] + j, 3 * atoms[i]: 3 * atoms[i] + 3] += deriv2[i][j][i]
        self.h_matrix = np.zeros(
            (len(self.ic), 3 * self.len, 3 * self.len), float)
        self.h_matrix = tmp_h_matrix

    def get_coordinates(self, atoms):
        """private method to retrive atoms' cartesian coordinates

        Arguments:
            atoms (tuple): a tuple of atoms indexes

        Returns:
            numpy.array, shape(N, 3); the N*3 coordinates of input atoms
        """
        coordinates = np.array([])
        for i in atoms:
            coordinates = np.append(coordinates, self.coordinates[i])
        return coordinates.reshape(-1, 3)

    def _get_difference(self):
        """getter of self.ic_differences. To calculate ic difference between present ic value and target ic value.
        return a list.
        """
        if len(self._ic_differences) < len(self.ic):
            self._ic_differences = np.append(self._ic_differences, np.zeros(
                (len(self.ic) - len(self._ic_differences)), float))
        return self.ic - self.target_ic

    def _set_difference(self, value):
        """setter for self.target. Use the value of difference and present ic to calculate target ic value.

        Arguments:
            value (numpy.array): An array of difference between actual coordinates and target coordinates
        """
        value = np.array(value)
        if len(value) <= len(self.ic):
            value = np.append(value, np.zeros(
                (len(self.ic) - len(value)), float))
        else:
            raise AtomsNumberError
        self.target_ic = self.ic - value

    ic_differences = property(_get_difference, _set_difference)

    def _get_target_ic(self):
        """getter for self.target.
        """
        if len(self._target_ic) < len(self.ic):
            self._target_ic = np.append(self._target_ic, self.ic[len(self._target_ic):])
        return self._target_ic

    def _set_target_ic(self, value):
        """setter for self.target

        Arguments:
            value (numpy.array): target ic value

        Raises:
            AtomsNumberError: number target ic is larger than actual ic
        """
        value = np.array(value)
        if len(value) <= len(self.ic):
            self._target_ic = np.append(value, self.ic[len(value):])
        else:
            raise AtomsNumberError

    target_ic = property(_get_target_ic, _set_target_ic)

    @property
    def weight_matrix(self):
        """setter of a property method for get weight matrix for coordinates transformation.
        """

        _weight_matrix = np.zeros((len(self.ic), len(self.ic)), float)
        for i in range(len(self.procedures)):
            info = self.procedures[i][0]
            atoms = self.procedures[i][1]
            if info == "add_dihed_angle":
                ic_function = ICTransformation._IC_types["add_bend_angle"]
                coordinates1 = self.get_coordinates(
                    (atoms[0], atoms[1], atoms[2]))
                angle1 = ic_function(coordinates1, deriv=0)[0]
                coordinates2 = self.get_coordinates(
                    (atoms[1], atoms[2], atoms[3]))
                angle2 = ic_function(coordinates2, deriv=0)[0]
                _weight_matrix[i][i] = np.sin(angle1)**2 * np.sin(angle2)**2
            else:
                _weight_matrix[i][i] = 1
        return _weight_matrix

    def calculate_cost(self):
        """function to calculate Cost which is used measure the difference between present ic with target one.

        Return:
            (float, numpy.array, numpy.array)
            float: the value of Cost function
            numpy.array, shape(3N,): the first derivative of cost function to cartesian coordinates
            numpy.array, shape(3N, 3N); the second derivative of cost function to cartesian coordinates
        """

        c_value = self._calculate_cost_value()
        c_x_deriv_1, c_x_deriv_2 = self._calculate_cost_deriv()
        return c_value, c_x_deriv_1, c_x_deriv_2

    def _calculate_cost_value(self):
        """fnnction to calculate cost function value.

        Return:
            float; the value of cost function
        """

        c_value = 0
        ic_num = len(self.ic)
        for i in range(ic_num):
            func = ICTransformation._IC_costs_value[self.procedures[i][0]]
            origin = self.ic[i]
            target = self.target_ic[i]
            c_value += func(origin, target) * self.weight_matrix[i][i]
        return c_value

    def _calculate_cost_deriv(self):
        """fnnction to calculate cost function value.

        Return:
         | ``c_x_deriv_1`` Numpy.array, shape(3N,); the first derivative of cost function to cartesian coordinates
         | ``c_x_deriv_2`` Numpy.array, shape(3N, 3N); the second derivative of cost function to cartesian coordinates
        """
        ic_num = len(self.ic)
        c_deriv = np.zeros(ic_num, float)
        c_deriv_2 = np.zeros((ic_num, ic_num), float)
        for i in range(ic_num):
            deriv_func = ICTransformation._IC_costs_diff[self.procedures[i][0]]
            deriv_2_func = ICTransformation._IC_costs_diff_2[
                self.procedures[i][0]]
            origin = self.ic[i]
            target = self.target_ic[i]
            c_deriv[i] += deriv_func(origin, target) * self.weight_matrix[i][i]
            c_deriv_2[i][
                i] += deriv_2_func(origin, target) * self.weight_matrix[i][i]
        c_x_deriv = np.dot(c_deriv, self.b_matrix)
        c_x_deriv_2_part_1 = np.dot(
            np.dot(np.transpose(self.b_matrix), c_deriv_2), self.b_matrix)
        c_x_deriv_2_part_2 = np.tensordot(c_deriv, self.h_matrix, 1)
        c_x_deriv_2 = c_x_deriv_2_part_1 + c_x_deriv_2_part_2
        return c_x_deriv, c_x_deriv_2

    def generate_point_object(self):
        """generate a Ponit object with present molecule's coordinates, first derivative and second derivative
        Return: a Point object; contains coordinates, cost function value, first derivative and second derivative
        """

        value, deriv_1, deriv_2 = self.calculate_cost()
        return op.Point(self.coordinates.reshape(1, -1), value, deriv_1, deriv_2)

    def cost_func_value_api(self, point):
        """accept a Point object to update local value and update the information of that object

        Arguments:
            point (optimize.Point): optimizer.Point object

        Return:
            optimizer.Point object with cost function value updated
        """
        self.get_new_coor(point.coordinates.reshape(-1, 3))
        point.value = self._calculate_cost_value()
        return point

    def ic_swap(self, icindex1, icindex2):
        temp = self.procedures[icindex1]
        self.procedures[icindex1] = self.procedures[icindex2]
        self.procedures[icindex2] = temp
        self._target_ic_swap(icindex1, icindex2)
        self._reset_ic()

    def _target_ic_swap(self, icindex1, icindex2):
        target_ic = self.target_ic
        temp = self._target_ic[icindex1]
        self._target_ic[icindex1] = self._target_ic[icindex2]
        self._target_ic[icindex2] = temp

    def cost_func_deriv_api(self, point):
        """accept a Point object to update local value and update the information of that object

        Arguments:
            point (optimizer.Point object): optimizer.Point Object

        Return:
            optimizer.Point Object; Point object with first derivative and second derivative updated
        """
        point.first_deriv, point.second_deriv = self._calculate_cost_deriv()
        return point

    def get_new_coor(self, new_coor):
        """update internal coordinates according to new cartesian coordinates.

        Arguments:
            new_coor (numpy.array): shape(N, 3); updated new cartesian coordinates.

        Raises:
            AtomsNumberError: new cartesian coordinates is different from original coordinates
        """
        if new_coor.shape != self.coordinates.shape:
            raise AtomsNumberError
        self.coordinates = new_coor
        self._reset_ic()

    def _reset_ic(self):
        """private method to calculate each internal coordinates again to get the updated ic value for new coordinates
        """
        self.iteration_flag = True
        self.ic = np.array([])
        self.b_matrix = np.zeros((0, 3 * self.len), float)
        self.h_matrix = np.zeros((0, 3 * self.len, 3 * self.len), float)
        for i in self.procedures:
            self._add_ic(i[0], i[1])
        self.iteration_flag = False

    _IC_types = {
        "add_bond_length": ICFunctions.bond_length,
        "add_bend_angle": ICFunctions.bend_angle,
        "add_dihed_angle": ICFunctions.dihed_angle,
        "add_dihed_angle_new_dot": ICFunctions.dihed_angle_new_dot,
        "add_dihed_angle_new_cross": ICFunctions.dihed_angle_new_cross
    }

    _IC_costs_diff = {
        "add_bond_length": CostFunctions.direct_diff,
        "add_bend_angle": CostFunctions.cos_diff,
        "add_dihed_angle": CostFunctions.dihed_diff,
        "add_dihed_angle_new_dot": CostFunctions.direct_diff,
        "add_dihed_angle_new_cross": CostFunctions.direct_diff
    }

    _IC_costs_value = {
        "add_bond_length": CostFunctions.direct_square,
        "add_bend_angle": CostFunctions.cos_square,
        "add_dihed_angle": CostFunctions.dihed_square,
        "add_dihed_angle_new_dot": CostFunctions.direct_square,
        "add_dihed_angle_new_cross": CostFunctions.direct_square
    }

    _IC_costs_diff_2 = {
        "add_bond_length": CostFunctions.direct_diff_2,
        "add_bend_angle": CostFunctions.cos_diff_2,
        "add_dihed_angle": CostFunctions.dihed_diff_2,
        "add_dihed_angle_new_dot": CostFunctions.direct_diff_2,
        "add_dihed_angle_new_cross": CostFunctions.direct_diff_2
    }

class AtomsNumberError(Exception):
    pass

if __name__ == '__main__':
    import horton as ht
    fn_xyz = ht.context.get_fn("test/water.xyz")
    mol = ht.IOData.from_file(fn_xyz)
    h2a = ICTransformation(mol.coordinates)
    h2a.add_bond_length(0, 1)
    h2a.add_bond_length(1, 2)
    h2a.add_bond_length(2, 1)
    h2a.add_bend_angle(0,1,2)
    print h2a.angle_calculate(0,1,2)
    print h2a.ic_info
    print h2a.ic
    print h2a.procedures
    # h2a._set_target_ic([2.8, 2.6])
    print h2a._target_ic
    h2a.ic_swap(0, 1)
    print h2a.ic
    print h2a.procedures
    print h2a.coordinates
#     print h2a.ic
#     h2a.add_bond_length(0, 2)
#     h2a.add_bond_length(2, 5)
#     h2a.add_bond_length(1, 3)
#     h2a.add_bond_length(1, 4)
#     print h2a.len
#     print h2a.H_matrix.shape
#     h2a.add_bend_angle(0,2,1)
#     h2a.add_bend_angle(0,1,2)
#     h2a.add_bend_angle(0,2,5)
#     h2a.add_bend_angle(0,1,3)
#     h2a.add_bend_angle(0,1,4)
#     h2a.add_dihed_new(1,0,2,5)
#     h2a.add_dihed_angle(0,1,3,4)
#     # h2a.target_ic = [2.7, 2.5, 2.3, 2.0,1.8, 1.7,1.0, 0.7, 2.7]
#     print h2a.ic
#     # print h2a.target_ic
#     # print h2a.ic_differences
#     h2a.target_ic = [2.7, 2.5, 2.3, 2.0,1.8, 1.7,1.0, 0.7, 2.7, 1.8, 1.8, 0.5,0.5,-2.5]
#     # print h2a.target_ic
#     # a,b,c = h2a.calculate_cost()
#     # # print np.linalg.inv(c)
#     # print b
#     c = h2a.generate_point_object()
#     # firstP = Point(h2a.coordinates.reshape(1,-1), a, b, c)
#     oph2a = DOM(c, h2a, IC_Transformation.new_coor_acceptor)
#     # print h2a.coordinates
#     # print h2a.ic
#     oph2a.first_step()
#     # print h2a.coordinates
#     # print h2a.ic
#     oph2a.second_step()
#     print h2a.ic

#     fn_xyz = ht.context.get_fn("test/water.xyz")
#     mol = ht.IOData.from_file(fn_xyz)
#     print mol.coordinates
    # print water.numbers.shape
    # water.add_bond_length(0,1)
    # water.add_bond_length(1,2)
    # water.add_bend_angle(0,1,2)
    # print water.ic
    # water.target_ic = [1.5,1.5,1.8]
    # point_start = water.generate_point_object()
    # print point_start
    # point_dom = op.DOM.initialize(point_start)
    # point_dom_new = op.DOM.update(point_dom, water.cost_func_value_api, water.cost_func_deriv_api)
    # print point_dom_new.coordinates
    # point_dom_op = op.DOM.optimize(point_dom_new, water.cost_func_value_api, water.cost_func_deriv_api)
    # print point_dom_new.coordinates
    # print water.ic
    # print water.calculate_cost()
    # # print water.calculate_cost_deriv()[1]
    # # print water.calculate_cost()[2] == water.calculate_cost_deriv()[1]
    # # print water.calculate_cost_value() == water.calculate_cost()[0]

    # # print water.target_ic
    # # print water.ic_differences
    # # water.ic_differences = [0.1,0.1]
    # # print water.target_ic
    # # print water.ic_differences
    # # water.add_bend_angle(1,0,2)
    # # print water.ic
    # # print water.target_ic
    # # print water.ic_differences
    # print water.ic_differences
    # a,b,c = water.calculate_cost()
    # print b, "this is b"
    # firstP = Point(water.coordinates.reshape(1,-1), a, b, c)
    # opwater = DOM(firstP, water, IC_Transformation.new_coor_acceptor)
    # print opwater.p0.point.value
    # print "__"
    # opwater.first_step()
    # print water.coordinates
    # print water.ic
    # opwater.first_step()
    # print opwater.p1.stepratio
    # opwater.second_step()
    # print opwater.p1.point.coordinates
    # print np.dot(opwater.p1.point.first_deriv, opwater.p1.point.first_deriv)
    # print water.coordinates
    # print water.ic

    # print water.coordinates.shape
    # fn_xyz = ht.context.get_fn('test/2h-azirine.xyz')
    # mol = ht.IOData.from_file(fn_xyz)
    # cc_object = IC_Transformation(mol)
    # print cc_object.coordinates
    # print cc_object.numbers
    # cc_object.add_bond_length(0,1) ## 1st ic
    # print cc_object.ic
    # print cc_object.procedures[0][1]
    # cc_object.add_bond_length(0,1)
    # print cc_object.ic
    # cc_object.add_bend_angle(1,2,3) ## 2nd ic
    # print cc_object.ic
    # print cc_object.procedures
    # cc_object.add_dihed_angle(1,2,3,4) ## 3rd ic
    # print cc_object.ic
    # cc_object.add_dihed_new(4,3,2,1) ## 4th, 5th ic
    # # cc_object.add_dihed_new(1,2,3,4)
    # print cc_object.ic
    # print cc_object.bond
    # # cc_object.ic = []
    # # cc_object.B_matrix = B_matrix = np.zeros((0,3*cc_object.len),float)
    # # print cc_object.ic
    # # cc_object.iteration_flag = True
    # print cc_object.procedures
    # # for i in cc_object.procedures:
    # #     cc_object._add_ic(i[0], i[1])
    # #     print i
    # print cc_object.ic
    # print cc_object.ic_differences
    # cc_object.ic_differences = [0.2,0.2,0.1,0.0,0.0]
    # print cc_object.target_ic
    # print cc_object.ic_differences
    # # cc_object.add_bond_length(1,2) ## 6th ic
    # print cc_object.ic
    # print cc_object.ic_differences
    # # cc_object.add_bond_length(2,3)
    # print cc_object.ic
    # print cc_object.B_matrix
    # a,b,c = cc_object.calculate_cost()
    # print c.shape
    # print cc_object.H_matrix.shape
