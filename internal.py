from __future__ import absolute_import, print_function

import numpy as np

from horton import periodic
from saddle.abclass import CoordinateTypes
from saddle.cartesian import Cartesian
from saddle.coordinate_types import BendAngle, BondLength, ConventionDihedral
from saddle.cost_functions import direct_square
from saddle.errors import (AtomsIndexError, AtomsNumberError, NotConvergeError,
                           NotSetError)
from saddle.opt import GeoOptimizer, Point


class Internal(Cartesian):
    """Internal Coordinate

    Properties
    ----------
    numbers : np.ndarray(N)
        A numpy array of atomic number for input coordinates
    spin : int
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
    cost_value_in_cc : tuple(float, np.ndarray(K), np.ndarray(K, K))
        Return the cost function value, 1st, and 2nd
        derivative verse cartesian coordinates
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
        Jacobian matrix for transfomr from cartesian coordinates to internal
        coordinates
    internal_gradient : np.ndarray(K,)
        Gradient of energy versus internal coordinates

    Methods
    -------
    __init__(coordinates, numbers, charge, spin)
        Initializes molecule
    set_new_coordinates(new_coor)
        Set molecule with a set of coordinates
    energy_calculation(**kwargs)
        Calculate system energy with different methods through
        software like gaussian
    distance(index1, index2)
        Calculate distance between two atoms with index1 and index2
    angle(index1, index2, index3)
        Calculate angle between atoms with index1, index2, and index3
    add_bond(atom1, atom2)
        Add a bond between atom1 and atom2
    add_angle_cos(atom1, atom2, atom3)
        Add a cos of a angle consist of atom1, atom2, and atom3
    add_dihedral(atom1, atom2, atom3, atom4)
        Add a dihedral of plain consists of atom1, atom2, and atom3
        and the other one consist of atom2, atom3, and atom4
    set_target_ic(new_ic)
        Set a target internal coordinates to transform into
    converge_to_target_ic(iteration=100, copy=True)
        Implement optimization process to transform geometry to
        target internal coordinates
    wipe_ic_info(I_am_sure_i_am_going_to_wipe_all_ic_info)
        wipe internal coordinates information in this structure
    set_new_ics(new_ics)
        Set the internal coordinates depends on the given internal
        coordinates
    print_connectivity()
        print connectivity matrix information on the screen
    swap_internal_coordinates(index_1, index_2)
        swap the two internal coordinates sequence
    connected_indices(index)
        Return a list of indices that connected to given index
    energy_from_fchk(abs_path, gradient=True, hessian=True):
        Obtain energy and corresponding info from fchk file
    auto_select_ic(dihed_special=False)
        automatic internal coordinates depends on buildin algorithm
    """

    def __init__(self, coordinates, numbers, charge, spin):
        super(Internal, self).__init__(coordinates, numbers, charge, spin)
        self._ic = []
        # 1 is connected, 0 is not, -1 is itself
        self._connectivity = np.diag([-1] * len(self.numbers))
        self._target_ic = None
        self._cc_to_ic_gradient = None
        self._cc_to_ic_hessian = None
        self._internal_gradient = None
        self._internal_hessian = None
        # self._tilt_internal_hessian = None

    def add_bond(self, atom1, atom2):  # tested
        """Add bond connection between atom1 and atom2

        Arguments
        ---------
        atom1 : int
            the index of the first atom
        atom2 : int
            the index of the second atom
        """
        if atom1 == atom2:
            raise AtomsIndexError("The two indece are the same")
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
        """Add cos angle connection between atom1, atom2 and atom3. The angle
        is consist of vector(atom1 - atom2) and vector(atom3 - atom2)

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
        rs = self.coordinates[np.array(atoms)]
        new_ic_obj = BendAngle(atoms, rs)
        d, dd = new_ic_obj.get_gradient_hessian()
        # check if the angle is formed by two connected bonds
        if self._check_connectivity(atom1, atom2) and self._check_connectivity(
                atom2, atom3):
            if self._repeat_check(new_ic_obj):
                self._add_new_internal_coordinate(new_ic_obj, d, dd, atoms)

    def add_dihedral(self, atom1, atom2, atom3, atom4):  # tested
        """Add a dihedral connection for atom1, atom2, atom3, and atom4
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
        atoms = self._atoms_sequence_reorder(atoms)
        rs = self.coordinates[np.array(atoms)]
        new_ic_obj = ConventionDihedral(atoms, rs)
        d, dd = new_ic_obj.get_gradient_hessian()
        if (self._check_connectivity(atom2, atom3) and
            (self._check_connectivity(atom1, atom2) or
             self._check_connectivity(atom1, atom3)) and
            (self._check_connectivity(atom4, atom3) or
             self._check_connectivity(atom4, atom2))):
            if self._repeat_check(new_ic_obj):
                self._add_new_internal_coordinate(new_ic_obj, d, dd, atoms)

    def set_target_ic(self, new_ic):
        """Set a target internal coordinates to optimize

        Arguments
        ---------
        new_ic : np.ndarray(K,) or list of int, len(new_ic) = K
        """
        if len(new_ic) != len(self.ic):
            raise AtomsNumberError("The ic is not in the same shape")
        self._target_ic = np.array(new_ic).copy()

    def set_new_coordinates(self, new_coor):  # to be tested
        """Assign new cartesian coordinates to this molecule

        Arguments
        ---------
        new_coor : np.ndarray(N, 3)
            New cartesian coordinates of the system
        """
        super(Internal, self).set_new_coordinates(new_coor)
        self._regenerate_ic()

    def swap_internal_coordinates(self, index_1, index_2):
        """Swap two internal coordinates sequence

        Arguments
        ---------
        index_1 : int
            index of the first internal coordinate
        index_2 : int
            index of the second internal coordinate
        """
        self._ic[index_1], self._ic[index_2] = self._ic[index_2], self._ic[
            index_1]
        self._regenerate_ic()

    def converge_to_target_ic(self, iteration=100):  # to be test
        """Using buildin optimization process to optimize geometry to target
        internal coordinates as close as possible

        Arguments
        ---------
        iteration : int, iteration > 0, default is 100
            number of iteration for optimization process
        """
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
                print("finished")
                return
            optimizer.update_trust_radius(optimizer.newest)
        raise NotConvergeError("The optimization failed to converge")

    def connected_indices(self, index):
        """Return the indices of atoms connected to given index atom

        Arguments
        ---------
        index : int
            the index of given index for finding connection

        Returns
        -------
        connected_index : np.ndarray(M)
            indices of atoms connected to given index
        """
        connection = self.connectivity[index]
        connected_index = np.where(connection > 0)[0]
        return connected_index

    def energy_from_fchk(self, abs_path, gradient=True, hessian=True):
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
        self._energy_hessian_transformation()

    def energy_calculation(self, **kwargs):
        """Conduct calculation with designated method.

        Keywords Arguments
        ------------------
        title : str, default is 'untitled'
            title of input and out put name without postfix
        method : str, default is 'g09'
            name of the program(method) used to calculate energy and other
            property
        """
        super(Internal, self).energy_calculation(**kwargs)
        self._energy_hessian_transformation()
        # h_q = (B^T)^+ \cdot (H_x - K) \cdot B^+

    def wipe_ic_info(self, I_am_sure_i_am_going_to_wipe_all_ic_info):
        """wipe all internal coordinates information in this structure
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

    def set_new_ics(self, new_ics):
        assert all(isinstance(ic, CoordinateTypes) for ic in new_ics)
        self.wipe_ic_info(True)
        self._ic = list(new_ics)
        self._regenerate_ic()
        self._regenerate_connectivity()

    @property
    def cost_value_in_cc(self):
        """Cost function value and its gradient, hessian versus Cartesian
        coordinates

        Returns
        v, x_d, x_dd : tuple(float, np.ndarray(3N,), np.ndarray(3N, 3N))
            v, the value of cost function
            x_d, the gradient vs cartesian coordinates
            x_dd, the hessian vs cartesian coordinates
        """
        v, d, dd = self._cost_value()
        x_d, x_dd = self._ic_gradient_hessian_transform_to_cc(d, dd)
        return v, x_d, x_dd

    @property
    def ic(self):
        """list of internal coordinates object

        Returns
        -------
        ic : list of coordinate_types, len(ic) = K
        """
        return self._ic

    @property
    def ic_values(self):
        """list of internal coordinates values

        Returns
        -------
        ic_values : list of float, len(ic_values) = K
        """
        value = [i.value for i in self._ic]
        return np.array(value)

    @property
    def target_ic(self):
        """target internal coordinates values

        Returns
        -------
        target_ic : np.ndarray(K,)
        """
        return self._target_ic

    @property
    def connectivity(self):
        """A connectivity matrix shows the connection of atoms, 1 is
        connected, 0 is not connected, -1 is itself

        Returns
        -------
        connectivity : np.ndarray(K, K)
        """
        return self._connectivity

    @property
    def b_matrix(self):
        """Jacobian matrix for transforming cartisian coordinates into
        internal coordinates

        Returns
        -------
        b_matrix : np.ndarray(K, 3N)
        """
        return self._cc_to_ic_gradient

    @property
    def internal_gradient(self):
        """Gradient of energy versus internal coordinates

        Returns
        -------
        internal_gradient : np.ndarray(K,)
        """
        return self._internal_gradient

    def print_connectivity(self):
        """Print the connectivity matrix on screen
        """
        format_func = "{:3}".format
        print("--Connectivity Starts-- \n")
        for i in range(len(self.numbers)):
            print(" ".join(map(format_func, self.connectivity[i, :i + 1])))
            print("\n--Connectivity Ends--")

    def auto_select_ic(self, dihed_special=False):
        """A method for Automatically selecting internal coordinates based on
        out buildin algorithm

        Arguments
        ---------
        dihed_special : bool, default is False
            choice of special dihedral indicator for dealing with collinear problem.
            True for enable, otherwise False
        """
        self._auto_select_bond()
        self._auto_select_angle()
        self._auto_select_dihed_normal()
        self._auto_select_dihed_improper()

    def _clear_ic_info(self):  # tested
        """Wipe all the internal information in this structure
        """
        self._ic = []
        self._connectivity = np.diag([-1] * len(self.numbers))
        self._target_ic = None
        self._cc_to_ic_gradient = None
        self._cc_to_ic_hessian = None
        self._internal_gradient = None
        self._internal_hessian = None

    def _auto_select_bond(self):
        """A private method for automatically selecting bond
        """
        halidish_atom = set([7, 8, 9, 15, 16, 17])
        for index_i in range(len(self.numbers)):
            for index_j in range(index_i + 1, len(self.numbers)):
                atom_num1 = self.numbers[index_i]
                atom_num2 = self.numbers[index_j]
                distance = self.distance(index_i, index_j)
                radius_sum = periodic[atom_num1].cov_radius + periodic[
                    atom_num2].cov_radius
                if distance < 1.3 * radius_sum:
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
                            atom_num3 = self.numbers[index_k]
                            if atom_num3 in halidish_atom:
                                dis = self.distance(h_index, index_k)
                                angle = self.angle(halo_index, h_index,
                                                   index_k)
                                thresh_sum = periodic[self._numbers[
                                    h_index]].vdw_radius + \
                                    periodic[self._numbers[index_k]].vdw_radius
                                if dis <= 0.9 * thresh_sum and angle >= 1.5708:
                                    self.add_bond(h_index, index_k)

    def _auto_select_angle(self):
        """A private method for automatically selecting angle
        """
        for center_index in range(len(self.numbers)):
            connected = self.connected_indices(center_index)
            if len(connected) >= 2:
                for edge_1 in range(len(connected)):
                    for edge_2 in range(edge_1 + 1, len(connected)):
                        self.add_angle_cos(connected[edge_1], center_index,
                                           connected[edge_2])

    def _auto_select_dihed_normal(self):
        """A private method for automatically selecting normal dihedral
        """
        for center_ind_1 in range(len(self.numbers)):
            connected = self.connected_indices(center_ind_1)
            if len(connected) >= 2:
                for center_ind_2 in connected:
                    sum_cnct = np.sum(self.connectivity, axis=0)
                    sum_select_cnct = sum_cnct[connected]
                    sorted_index = sum_select_cnct.argsort()[::-1]
                    side_1 = connected[sorted_index[0]]
                    if connected[sorted_index[0]] == center_ind_2:
                        side_1 = connected[sorted_index[1]]
                    connected_to_index_2 = self.connected_indices(center_ind_2)
                    for side_2 in connected_to_index_2:
                        if side_2 not in (center_ind_1, center_ind_2, side_1):
                            self.add_dihedral(side_1, center_ind_1,
                                              center_ind_2, side_2)

    def _auto_select_dihed_improper(self):
        """A private method for automatically selecting improper dihedral
        """
        connect_sum = np.sum(self.connectivity, axis=0)
        for center_ind in range(len(connect_sum)):
            if connect_sum[center_ind] >= 3:
                cnct_atoms = self.connected_indices(center_ind)
                cnct_total = len(cnct_atoms)
                for i in range(cnct_total):
                    for j in range(i + 1, cnct_total):
                        for k in range(j + 1, cnct_total):
                            ind_i, ind_j, ind_k = cnct_atoms[[i, j, k]]
                            ang1_r = self.angle(ind_i, center_ind, ind_j)
                            ang2_r = self.angle(ind_i, center_ind, ind_k)
                            ang3_r = self.angle(ind_j, center_ind, ind_k)
                            sum_r = ang1_r + ang2_r + ang3_r
                            if sum_r >= 6.02139:
                                self.add_dihedral(ind_i, center_ind, ind_j,
                                                  ind_k)

    def _energy_hessian_transformation(self):
        """convert gradient, hessian versus cartesian coordinates into
        gradient, hessian versus internal coordinates
        ..math::
            g_q = (B_T)^+ g_x
            H_q = B_T^+ (H_x - K) B^+ + K, where
            K = g_q b^\prime
        """
        self._internal_gradient = np.dot(
            np.linalg.pinv(self._cc_to_ic_gradient.T), self._energy_gradient)
        # g_q = (B^T)^+ \cdot g_x
        hes_K = self._energy_hessian - np.tensordot(
            self._internal_gradient, self._cc_to_ic_hessian, axes=1)
        self._internal_hessian = np.dot(
            np.dot(np.linalg.pinv(self._cc_to_ic_gradient.T), hes_K),
            np.linalg.pinv(self._cc_to_ic_gradient))
        # self._tilt_internal_hessian = np.dot(
        #     np.dot(
        #         np.linalg.pinv(self._cc_to_ic_gradient.T),
        #         self._energy_hessian), np.linalg.pinv(self._cc_to_ic_gradient))

    def _regenerate_ic(self):
        """reset internal coordinates system, reset gradient, hessian versus
        internal coordinates, regenerate internal coordinates and
        transformation matrix
        """
        self._cc_to_ic_gradient = None
        self._cc_to_ic_hessian = None
        for ic in self.ic:
            rs = self.coordinates[np.array(ic.atoms)]
            ic.set_new_coordinates(rs)
            d, dd = ic.get_gradient_hessian()
            self._add_cc_to_ic_gradient(d, ic.atoms)  # add transform gradient
            self._add_cc_to_ic_hessian(dd, ic.atoms)  # add transform hessian
        self._recal_g_and_h()  # clean internal gradient and hessian

    def _regenerate_connectivity(self):
        """regenerate the connectivity of molecule depends on present
        internal coordinates
        """
        self._connectivity = np.diag([-1] * len(self.numbers))
        for ic in self.ic:
            if isinstance(ic, BondLength):
                self._add_connectivity(ic.atoms)

    def _recal_g_and_h(self):
        """reset internal energy gradient and hessian matrix
        """
        self._internal_gradient = None
        self._internal_hessian = None
        if (self._energy_gradient is not None and
                self._energy_hessian is not None):
            self._energy_hessian_transformation()

    def _create_geo_point(self):
        """create a Point object based on self internal coordinates to undergo
        a optimizatino process in order to converge to target_ic

        Returns
        -------
        geo_point : Point object
        """
        _, x_d, x_dd = self.cost_value_in_cc
        return Point(x_d, x_dd, len(self.numbers))

    def _cost_value(self):
        """Calculate value of cost function as well as its gradient and hessian versus internal coordinates

        Returns
        -------
        v, d, dd : tuple(float, np.ndarray(K,), np.ndarray(K, K))
            v, the value of cost function
            d, the gradient vs internal coordinates
            dd, the hessian vs internal coordinates
        """
        v, d, dd = self._calculate_cost_value()
        return v, d, dd

    def _calculate_cost_value(self):
        """a low level function to calculate cost function as well as its
        gradient and hessian

        Returns
        -------
        value, deriv, deriv2 : tuple(float, np.ndarray(K,), np.ndarray(K, K))
            value, the value of cost function
            deriv, the gradient vs internal coordinates
            deriv2, the hessian vs internal coordinates
        """
        if self.target_ic is None:
            raise NotSetError("The value of target_ic is not set")
        # initialize function value, gradient and hessian
        value = 0
        deriv = np.zeros(len(self.ic))
        deriv2 = np.zeros((len(self.ic), len(self.ic)), float)
        for i in range(len(self.ic)):
            if self.ic[i].__class__.__name__ in ("BondLength",
                                                 "BendAngle", ):
                v, d, dd = direct_square(self.ic_values[i], self.target_ic[i])
                value += v
                deriv[i] += d
                deriv2[i, i] += dd
        return value, deriv, deriv2

    def _ic_gradient_hessian_transform_to_cc(self, gradient, hessian):
        """transform energy gradient and hessian back from internal coordinates to
        cartesian cooridnates
        ..math::
            g_x = B_T g_q
            H_x = B_T H_q B + K, where
            K = g_q b^\prime

        Returns
        -------
        cartesian_gradient, cartesian_hessian : tuple(np.ndarray(3N,),
                                                      np.ndarray(3N, 3N))
            cartesian_gradient, energy gradient vs cartesian coordinates
            cartesian_hessian, energy hessian vs cartesian coordinates
        """
        cartesian_gradient = np.dot(gradient, self._cc_to_ic_gradient)
        cartesian_hessian_part_1 = np.dot(
            np.dot(self._cc_to_ic_gradient.T, hessian),
            self._cc_to_ic_gradient)
        cartesian_hessian_part_2 = np.tensordot(gradient,
                                                self._cc_to_ic_hessian, 1)
        cartesian_hessian = cartesian_hessian_part_1 + cartesian_hessian_part_2
        return cartesian_gradient, cartesian_hessian

    def _check_connectivity(self, atom1, atom2):
        """Check whether two atoms are connected or not

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
        if self.connectivity[atom1, atom2] == 1:
            return True
        elif self.connectivity[atom1, atom2] == 0:
            return False

    def _repeat_check(self, ic_obj):
        """Check whether the given ic_obj already existed in ic list or not

        Arguments
        ---------
        ic_obj : Coordinate_Types
            the given ic object to be tested

        Returns
        -------
        repeat_check : bool
            Return True if there is no duplicate and it's a valid new ic
            object, otherwise False
        """
        for ic in self.ic:
            if ic_obj.atoms == ic.atoms and type(ic_obj) == type(ic):
                return False
        else:
            return True

    def _add_new_internal_coordinate(self, new_ic, d, dd, atoms):
        """Add a new ic object to the system and add corresponding
        transformation matrix parts
        """
        self._ic.append(new_ic)
        self._add_cc_to_ic_gradient(d, atoms)  # add gradient
        self._add_cc_to_ic_hessian(dd, atoms)  # add hessian
        self._recal_g_and_h()

    def _add_connectivity(self, atoms):
        """Change the value of connectivity matrix to 1 for two atoms

        Arguments
        ---------
        atoms : list or tuple of int, len(atoms) = 2
        """
        if len(atoms) != 2:
            raise AtomsNumberError("The number of atoms is not correct")
        num1, num2 = atoms
        self._connectivity[num1, num2] = 1
        self._connectivity[num2, num1] = 1

    def _atoms_sequence_reorder(self, atoms):
        """Change the atoms in each ic object in ascending sequence without
        changing its representative

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
            if atoms[0] > atoms[3]:
                atoms[0], atoms[3] = atoms[3], atoms[0]
            if atoms[1] > atoms[2]:
                atoms[1], atoms[2] = atoms[2], atoms[1]
        else:
            raise AtomsNumberError("The number of atoms is not correct")
        return tuple(atoms)

    def _add_cc_to_ic_gradient(self, deriv, atoms):  # need to be tested
        """Add new entries from a new ic to transformation matrix gradient

        Arguments
        ---------
        deriv : np.ndarray(3N,)
            tranformation gradient matrix given ic regarding to given atoms
        atoms : list or tuple of int
            indices of atoms for those transformation
        """
        if self._cc_to_ic_gradient is None:
            self._cc_to_ic_gradient = np.zeros((0, 3 * len(self.numbers)))
        tmp_vector = np.zeros((1, 3 * len(self.numbers)))
        for i in range(len(atoms)):
            tmp_vector[0, 3 * atoms[i]:3 * atoms[i] + 3] += deriv[i]
        self._cc_to_ic_gradient = np.vstack(
            (self._cc_to_ic_gradient, tmp_vector))

    def _add_cc_to_ic_hessian(self, deriv, atoms):  # need to be tested
        """Add new entries from a new ic to transformation matrix hessian

        Arguments
        ---------
        deriv : np.ndarray(3N, 3N)
            tranformation hessian matrix given ic regarding to given atoms
        atoms : list or tuple of int
            indices of atoms for those transformation
        """
        if self._cc_to_ic_hessian is None:
            self._cc_to_ic_hessian = np.zeros(
                (0, 3 * len(self.numbers), 3 * len(self.numbers)))
        tmp_vector = np.zeros(
            (1, 3 * len(self.numbers), 3 * len(self.numbers)))
        for i in range(len(atoms)):
            for j in range(len(atoms)):
                tmp_vector[0, 3 * atoms[i]:3 * atoms[i] + 3, 3 * atoms[j]:3 *
                           atoms[j] + 3] += deriv[i, :3, j]
        self._cc_to_ic_hessian = np.vstack(
            (self._cc_to_ic_hessian, tmp_vector))
