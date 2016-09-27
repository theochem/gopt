from __future__ import print_function, absolute_import
from saddle.cartesian import Cartesian
from saddle.errors import NotSetError, AtomsNumberError, NotConvergeError
from saddle.coordinate_types import BondLength, BendAngle, ConventionDihedral
from saddle.cost_functions import direct_square
from saddle.opt import GeoOptimizer, Point
from copy import deepcopy
import numpy as np



class Internal(Cartesian):
    """ Cartesian Coordinate

    Properties
    ----------
    numbers : np.ndarray(K)
        A list of atomic number for input coordinates
    spin : int
        Spin multiplicity of the molecule
    charge : int
        Charge of the input molecule
    energy : float
        Energy of given Cartesian coordinates system molecule
    energy_gradient : np.ndarray(K)
        Gradient of Energy that calculated through certain method
    energy_hessian : np.ndarray(K, K)
        Hessian of Energy that calculated through cartain method
    coordinates : np.ndarray(K, 3)
        Cartesian information of input molecule
    cost_value_in_cc : tuple(float, np.ndarray(K), np.ndarray(K, K))
        Return the cost function value, 1st, and 2nd derivative verse cartesian coordinates
    ic : list[CoordinateTypes]
        A list of CoordinateTypes instance to represent internal coordinates information
    ic_values : list[float]
        A list of internal coordinates values
    target_ic : list[float]
        A list of target internal coordinates
    connectivity : np.ndarray(K, K)
        A square matrix represents the connectivity of molecule internal coordinates

    Methods
    -------
    __init__(self, coordinates, numbers, charge, spin)
        Initializes molecule
    set_new_coordinates(new_coor)
        Set molecule with a set of coordinates
    energy_calculation(**kwargs)
        Calculate system energy with different methods through software like gaussian
    distance(index1, index2)
        Calculate distance between two atoms with index1 and index2
    angle(index1, index2, index3)
        Calculate angle between atoms with index1, index2, and index3
    add_bond(atom1, atom2)
        Add a bond between atom1 and atom2
    add_angle_cos(atom1, atom2, atom3)
        Add a cos of a angle consist of atom1, atom2, and atom3
    add_dihedral(atom1, atom2, atom3, atom4)
        Add a dihedral of plain consists of atom1, atom2, and atom3 and the other one consist of atom2, atom3, and atom4
    set_target_ic(new_ic)
        Set a target internal coordinates to transform into
    converge_to_target_ic(iteration=100, copy=True)
        Implement optimization process to transform geometry to target internal coordinates
    print_connectivity()
        print connectivity matrix information on the screen
    """
    def __init__(self, coordinates, numbers, charge, spin):
        super(Internal, self).__init__(coordinates, numbers, charge, spin)
        self._ic = []  # type np.array([float 64])
        # 1 is connected, 0 is not, -1 is itself
        self._connectivity = np.diag([-1] * len(self.numbers))
        self._target_ic = None
        self._cc_to_ic_gradient = None
        self._cc_to_ic_hessian = None

    def add_bond(self, atom1, atom2):  # tested
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

    def add_angle_cos(self, atom1, atom2, atom3):  # tested
        atoms = (atom1, atom2, atom3)
        atoms = self._atoms_sequence_reorder(atoms)
        rs = self.coordinates[np.array(atoms)]
        new_ic_obj = BendAngle(atoms, rs)
        d, dd = new_ic_obj.get_gradient_hessian()
        # check if the angle is formed by two connected bonds
        if self._check_connectivity(atom1, atom2) and self._check_connectivity(atom2, atom3):
            if self._repeat_check(new_ic_obj):
                self._add_new_internal_coordinate(new_ic_obj, d, dd, atoms)

    def add_dihedral(self, atom1, atom2, atom3, atom4):  # tested
        atoms = (atom1, atom2, atom3, atom4)
        atoms = self._atoms_sequence_reorder(atoms)
        rs = self.coordinates[np.array(atoms)]
        new_ic_obj = ConventionDihedral(atoms, rs)
        d, dd = new_ic_obj.get_gradient_hessian()
        if self._check_connectivity(atom2, atom3) and (self._check_connectivity(atom1, atom2) or self._check_connectivity(atom1, atom3)) and (self._check_connectivity(atom4, atom3) or self._check_connectivity(atom4, atom2)):
            if self._repeat_check(new_ic_obj):
                self._add_new_internal_coordinate(new_ic_obj, d, dd, atoms)

    def set_target_ic(self, new_ic):
        if len(new_ic) != len(self.ic):
            raise AtomsNumberError, "The ic is not in the same shape"
        self._target_ic = np.array(new_ic)

    def set_new_coordinates(self, new_coor): # to be tested
        super(Internal, self).set_new_coordinates(new_coor)
        self._cc_to_ic_gradient = None
        self._cc_to_ic_hessian = None
        for i in self.ic:
            rs = self.coordinates[np.array(i.atoms)]
            i.set_new_coordinates(rs)
            d, dd = i.get_gradient_hessian()
            self._add_cc_to_ic_gradient(d, i.atoms)  # add gradient
            self._add_cc_to_ic_hessian(dd, i.atoms)  # add hessian

    def converge_to_target_ic(self, iteration=100, copy=True):  # to be test
        if copy:
            self = deepcopy(self)
        optimizer = GeoOptimizer()
        init_point = self._create_geo_point()
        optimizer.add_new(init_point)
        for i in range(iteration):
            optimizer.tweak_hessian(optimizer.newest)
            step = optimizer.trust_radius_step(optimizer.newest)
            new_coor = self.coordinates + step.reshape(-1, 3)
            self.set_new_coordinates(new_coor)
            new_point = self._create_geo_point()
            optimizer.add_new(new_point)
            if optimizer.converge(optimizer.newest):
                print ("finished")
                return self
            optimizer.update_trust_radius(optimizer.newest)
        raise NotConvergeError, "The optimization failed to converge"

    def _create_geo_point(self):
        _, x_d, x_dd = self.cost_value_in_cc
        return Point(x_d, x_dd, len(self.numbers))

    def _cost_value(self):
        v, d, dd = self._calculate_cost_value()
        return v, d, dd
        # x_d, x_dd = self._ic_gradient_hessian_transform_to_cc(d, dd)
        # return v, x_d, x_dd

    @property
    def cost_value_in_cc(self):
        v, d, dd = self._cost_value()
        x_d, x_dd = self._ic_gradient_hessian_transform_to_cc(d, dd)
        return v, x_d, x_dd


    def _calculate_cost_value(self):
        if self.target_ic is None:
            raise NotSetError, "The value of target_ic is not set"
        # initialize function value, gradient and hessian
        value = 0
        deriv = np.zeros(len(self.ic))
        deriv2 = np.zeros((len(self.ic), len(self.ic)), float)
        for i in range(len(self.ic)):
            if self.ic[i].__class__.__name__ in ("BondLength", "BendAngle",):
                v, d, dd = direct_square(self.ic_values[i], self.target_ic[i])
                value += v
                deriv[i] += d
                deriv2[i, i] += dd
        return value, deriv, deriv2

    def _ic_gradient_hessian_transform_to_cc(self, gradient, hessian):
        cartesian_gradient = np.dot(gradient, self._cc_to_ic_gradient)
        cartesian_hessian_part_1 = np.dot(
            np.dot(self._cc_to_ic_gradient.T, hessian), self._cc_to_ic_gradient)
        cartesian_hessian_part_2 = np.tensordot(
            gradient, self._cc_to_ic_hessian, 1)
        cartesian_hessian = cartesian_hessian_part_1 + cartesian_hessian_part_2
        return cartesian_gradient, cartesian_hessian

    def _check_connectivity(self, atom1, atom2):
        if self.connectivity[atom1, atom2] == 1:
            return True
        elif self.connectivity[atom1, atom2] == 0:
            return False

    def _repeat_check(self, ic_obj):
        for ic in self.ic:
            if ic_obj.atoms == ic.atoms and type(ic_obj) == type(ic):
                return False
        else:
            return True

    def _add_new_internal_coordinate(self, new_ic, d, dd, atoms):
        self._ic.append(new_ic)
        self._add_cc_to_ic_gradient(d, atoms)  # add gradient
        self._add_cc_to_ic_hessian(dd, atoms)  # add hessian

    def _add_connectivity(self, atoms):
        if len(atoms) != 2:
            raise AtomsNumberError, "The number of atoms is not correct"
        num1, num2 = atoms
        self._connectivity[num1, num2] = 1
        self._connectivity[num2, num1] = 1

    def _atoms_sequence_reorder(self, atoms):
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
            raise AtomsNumberError, "The number of atoms is not correct"
        return tuple(atoms)

    def _add_cc_to_ic_gradient(self, deriv, atoms):  # need to be tested
        if self._cc_to_ic_gradient is None:
            self._cc_to_ic_gradient = np.zeros((0, 3 * len(self.numbers)))
        tmp_vector = np.zeros((1, 3 * len(self.numbers)))
        for i in range(len(atoms)):
            tmp_vector[0, 3 * atoms[i]: 3 * atoms[i] + 3] += deriv[i]
        self._cc_to_ic_gradient = np.vstack(
            (self._cc_to_ic_gradient, tmp_vector))

    def _add_cc_to_ic_hessian(self, deriv, atoms):  # need to be tested
        if self._cc_to_ic_hessian is None:
            self._cc_to_ic_hessian = np.zeros(
                (0, 3 * len(self.numbers), 3 * len(self.numbers)))
        tmp_vector = np.zeros(
            (1, 3 * len(self.numbers), 3 * len(self.numbers)))
        for i in range(len(atoms)):
            for j in range(len(atoms)):
                tmp_vector[0, 3 * atoms[i]: 3 * atoms[i] + 3, 3 *
                           atoms[j]: 3 * atoms[j] + 3] += deriv[i, :3, j]
        self._cc_to_ic_hessian = np.vstack(
            (self._cc_to_ic_hessian, tmp_vector))

    @property
    def ic(self):
        return self._ic

    @property
    def ic_values(self):
        return [i.value for i in self._ic]

    @property
    def target_ic(self):
        return self._target_ic

    @property
    def connectivity(self):
        return self._connectivity

    def print_connectivity(self):
        format_func = "{:3}".format
        print ("--Connectivity Starts-- \n")
        for i in range(len(self.numbers)):
            print (" ".join(map(format_func, self.connectivity[i, :i + 1])))
        print ("\n--Connectivity Ends--")


# import horton as ht
# fn_xyz = ht.context.get_fn("test/water.xyz")
# mol = ht.IOData.from_file(fn_xyz)  # create a water molecule
# inter = Internal(mol.coordinates, mol.numbers, 0, 1)
# inter.add_bond(0,2)
# inter.add_bond(0,1)
# print (inter._cc_to_ic_hessian.shape)
