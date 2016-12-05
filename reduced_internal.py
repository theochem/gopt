from __future__ import absolute_import, print_function

import numpy as np

from saddle.errors import NotSetError
from saddle.internal import Internal
from saddle.solver import diagonalize


class ReducedInternal(Internal):  # need tests
    """Reduced Internal Coordinate

    Properties
    ----------
    df : int
        Degree of freedom of a system, 3N - 5 for linear and 3N -5 for the
        rest
    key_ic_number : int
        Number of key internal coordinates which correspond to important
        chemical property
    vspace : np.ndarray(K, 3N - 6)
        Transformation matrix from internal coordinates to reduced internal
        coordinates
    vspace_gradient : np.ndarray(3N - 6,)
        Energy Gradient of system versus reduced internal coordinates
    vspace_hessian : np.ndarray(3N - 6, 3N - 6)
        Energy Hessian of system versus reduced internal coordinates
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
    set_key_ic_number(number)
        Set the number of key internal coordinates
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
    print_connectivity()
        Rrint connectivity matrix information on the screen
    swap_internal_coordinates(index_1, index_2)
        Swap the two internal coordinates sequence
    connected_indices(index)
        Return a list of indices that connected to given index
    energy_from_fchk(abs_path, gradient=True, hessian=True):
        Obtain energy and corresponding info from fchk file
    auto_select_ic(dihed_special=False)
        Automatic internal coordinates depends on buildin algorithm
    align_vspace(target)
        Regenerate a vspace transformation matrix to align with target's
        vspace
    set_vspace(new_vspace)
        Set the vspace matrix of system with given one
    update_to_new_structure_with_delta_v(delta_v)
        Calculate the new internal coordinates value depends on given change
        delta_v in V space

    Class Methods
    -------------
    update_to_reduced_internal(internal_ob, key_ic_number=0)
        update a InternalCoordinates into a ReducedInternalCoordinates object
    """

    def __init__(self, coordinates, numbers, charge, spin, key_ic_number=0):
        super(ReducedInternal, self).__init__(coordinates, numbers, charge,
                                              spin)
        self._k_ic_n = key_ic_number
        self._red_space = None
        self._non_red_space = None
        self._vspace_gradient = None
        self._vspace_hessian = None
        self._vspace = None

    @property
    def df(self):
        """The degree of the system

        Returns
        -------
        df : int
        """
        return len(self.numbers) * 3 - 6

    @property
    def key_ic_number(self):
        """The number of key internal coordinates

        Returns
        -------
        key_ic_number : int
        """
        return self._k_ic_n

    @property
    def vspace(self):
        """Vspace transformation matrix from internal to reduced internal

        Returns
        -------
        vspace : np.ndarray(K, 3N - 6)
        """
        if self._red_space is None or self._non_red_space is None:
            self._generate_reduce_space()
            self._generate_nonreduce_space()
            self._vspace = np.hstack((self._red_space, self._non_red_space))
        return self._vspace

    @property
    def vspace_gradient(self):
        """Gradient of energy versus reduced internal coordinates

        Returns
        -------
        vspace_gradient : np.ndarray(3N - 6,)
        """
        if self._vspace_gradient is None:
            self._vspace_gradient = np.dot(self.vspace.T,
                                           self._internal_gradient)
        return self._vspace_gradient

    @property
    def vspace_hessian(self):
        """Hessian of energy versus reduced internal coordinates

        Returns
        -------
        vspace_hessian : np.ndarray(3N - 6, 3N - 6)
        """
        if self._vspace_hessian is None:
            self._vspace_hessian = np.dot(
                np.dot(self.vspace.T, self._internal_hessian), self.vspace)
        return self._vspace_hessian

    def set_key_ic_number(self, number):
        """Set the value of key_ic_number of the system

        Arguments
        ---------
        number : int
            The value of key_ic_number going to be set
        """
        self._k_ic_n = number
        self._reset_v_space()

    @classmethod
    def update_to_reduced_internal(cls, internal_ob, key_ic_number=0):
        """Update a internal coordinates object into reduced internal
        coordinates object

        Arguments
        ---------
        internal_ob : InternalCoordinates
            Target InternalCooridnates object
        key_ic_number : int, default is 0
            The value of key internal coordinates
        """
        assert isinstance(internal_ob, Internal)
        internal_ob.__class__ = cls
        internal_ob._k_ic_n = key_ic_number
        internal_ob._reset_v_space()

    def align_vspace(self, target):
        """Align vspace with a given target ReducedInternal object

        Arguments
        ---------
        target : ReducedInternal
            the target ReducedInternal object to align to
        """
        assert isinstance(target, ReducedInternal)
        overlap = np.dot(self.vspace.T, target.vspace)
        u, s, v = np.linalg.svd(overlap)
        q_min = np.dot(u, v)
        new_v = np.dot(self.vspace, q_min)
        self.set_vspace(new_v)

    def set_vspace(self, new_vspace):
        """Set vspace of system with given values

        Arguments
        ---------
        new_vspace : np.ndarray(K, 3N - 6)
            The new value of vspace
        """
        self._vspace = new_vspace
        self._vspace_gradient = None
        self._vspace_hessian = None

    def set_new_coordinates(self, new_coor):
        """Assign new cartesian coordinates to this molecule

        Arguments
        ---------
        new_coor : np.ndarray(N, 3)
            New cartesian coordinates of the system
        """
        super(ReducedInternal, self).set_new_coordinates(new_coor)
        self._reset_v_space()

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
        super(ReducedInternal, self).energy_from_fchk(abs_path, gradient,
                                                      hessian)

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
        super(ReducedInternal, self).energy_calculation(**kwargs)

    def swap_internal_coordinates(self, index_1, index_2):
        """Swap two internal coordinates sequence

        Arguments
        ---------
        index_1 : int
            index of the first internal coordinate
        index_2 : int
            index of the second internal coordinate
        """
        super(ReducedInternal, self).swap_internal_coordinates(index_1,
                                                               index_2)
        self._reset_v_space()

    def update_to_new_structure_with_delta_v(self, delta_v):
        """Update system to a new internal coordinates structure given a
        change in vspace delta_v

        Arguments
        ---------
        delta_v : np.ndarray(3N - 6,)
            coordinates changes in vspace
        """
        delta_ic = self._get_delta_ic_from_delta_v(delta_v)
        new_ic = delta_ic + self.ic_values
        self.set_target_ic(new_ic)
        self.converge_to_target_ic()
        self._reset_v_space()

    def _get_delta_ic_from_delta_v(self, delta_v):
        """Calculate corresponding change in internal coordinates given a
        change in vspace coordinates

        Arguments
        ---------
        delta_v : np.ndarray(3N - 6,)
            The changes of coordinates in vspace coordinates

        Returns
        -------
        delta_ic : np.ndarray(K,)
            The changes of cooridnates in internal cooridnates
        """
        return np.dot(self.vspace, delta_v)

    def _add_new_internal_coordinate(self, new_ic, d, dd, atoms):  # add reset
        """Add a new ic object to the system and add corresponding
        transformation matrix parts
        """
        super(ReducedInternal, self)._add_new_internal_coordinate(new_ic, d,
                                                                  dd, atoms)
        self._reset_v_space()

    def _reset_v_space(self):
        """Reset vspace coordinates data, including vspace, gradient, hessian
        """
        self._red_space = None
        self._non_red_space = None
        self._vspace_gradient = None
        self._vspace_hessian = None
        self._vspace = None

    def _svd_of_cc_to_ic_gradient(self, threshold=1e-6):  # tested
        """Select 3N - 6 non-singular vectors from b_matrix through SVD

        Returns:
        u : np.ndarray(K, 3N - 6)
            3N - 6 non-singular vectors from SVD
        """
        u, s, v = np.linalg.svd(self._cc_to_ic_gradient, full_matrices=0)
        return u[:, np.abs(s) > threshold][:, :self.df]

    def _reduced_unit_vectors(self):  # tested
        """Generate unit vectors where every position is 0 except the
        key_ic_number position is 1

        Returns
        -------
        unit_mtx : np.ndarray(K, J), J is the number of key_ic_number
            Unit vectors with 0 everywhere except that key_ic_number position
            is 0
        """
        unit_mtx = np.zeros((len(self.ic), self.key_ic_number))
        unit_mtx[:self._k_ic_n, :self._k_ic_n] = np.eye(self._k_ic_n)
        return unit_mtx

    def _reduced_perturbation(self):  # tested
        """Calculate the realizable purterbation in internal cooridnates from
        the unit vectors

        Returns
        -------
        purterbs : np.ndarray(K, J)
            Realizable purterbation in internal cooridnates from those unit
            vectors
        """
        unit_mtx = self._reduced_unit_vectors()
        tsfm = np.dot(self._cc_to_ic_gradient,
                      np.linalg.pinv(self._cc_to_ic_gradient))
        return np.dot(tsfm, unit_mtx)

    def _generate_reduce_space(self, threshold=1e-6):  # tested
        """Generate reduced part of vspace matrix
        ..math::
            to be done

        Arguments
        ---------
        threshold : float, default is 1e-6
            the cutoff for 0 eigenvalues
        """
        b_mtx = self._reduced_perturbation()
        w, v = diagonalize(b_mtx)
        self._red_space = v[:, abs(w) > threshold]

    def _nonreduce_vectors(self):
        """Calculate the nonspace of reduced space

        Returns
        -------
        non_reduce_vectors : np.ndarray(K, 3N - 6 - J)
            the nonspace of reduced internal coordinates space
        """
        a_mtx = self._svd_of_cc_to_ic_gradient()
        rd_space = self._red_space
        prj_rd_space = np.dot(rd_space, rd_space.T)  # prj = \ket{\v} \bra{\v}
        non_reduce_vectors = a_mtx - np.dot(prj_rd_space, a_mtx)
        return non_reduce_vectors

    def _generate_nonreduce_space(self, threshold=1e-6):  # tested
        """Generate nonreduce part of vpsace matrix
        """
        d_mtx = self._nonreduce_vectors()
        w, v = diagonalize(d_mtx)
        self._non_red_space = v[:, abs(w) > threshold][:, :self.df - len(
            self._red_space[0])]
