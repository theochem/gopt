from copy import deepcopy

import numpy as np
from saddle.errors import ICNumberError, NotSetError
from saddle.internal import Internal
from saddle.math_lib import diagonalize, pse_inv

__all__ = ('ReducedInternal', )


class ReducedInternal(Internal):  # need tests
    """Reduced Internal Coordinate

    Properties
    ----------
    df : int
        Degree of freedom of a system, 3N - 5 for linear and 3N - 6 for the
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
    numbers : np.ndarray(N,)
        A numpy array of atomic number for input coordinates
    multi : int
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
    natom : int
        Number of atoms in the system
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
    __init__(coordinates, numbers, charge, multi)
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
    from_file(filename, charge=0, multi=1)
        Create cartesian instance from file
    """

    def __init__(self,
                 coordinates: 'np.ndarray',
                 numbers: 'np.ndarray',
                 charge: int,
                 multi: int,
                 title: str = "",
                 key_ic_number: int = 0) -> None:
        super(ReducedInternal, self).__init__(coordinates, numbers, charge,
                                              multi, title)
        self._k_ic_n = key_ic_number
        self._reset_v_space()

    @property
    def key_ic(self):
        """Return the key internal coordinates
        """
        return self.ic[:self.key_ic_number]

    @property
    def key_ic_number(self) -> int:
        """The number of key internal coordinates

        Returns
        -------
        key_ic_number : int
        """
        return self._k_ic_n

    @property
    def vspace(self) -> 'np.ndarray':
        """Vspace transformation matrix from internal to reduced internal

        Returns
        -------
        vspace : np.ndarray(K, 3N - 6)
        """
        if self.df > len(self.ic):
            raise ICNumberError(
                f'''Internal coordinates number {len(self.ic)} is less
                than the degree freedom of the system {self.df}''')
        if self._red_space is None or self._non_red_space is None:
            self._generate_reduce_space()
            self._generate_nonreduce_space()
            self._vspace = np.hstack((self._red_space, self._non_red_space))
        return self._vspace

    @property
    def vspace_gradient(self) -> 'np.ndarray':
        """Gradient of energy versus reduced internal coordinates

        Returns
        -------
        vspace_gradient : np.ndarray(3N - 6,)
        """
        if self.internal_gradient is None:
            raise NotSetError
        return np.dot(self.vspace.T, self.internal_gradient)

    v_gradient = vspace_gradient

    @property
    def vspace_hessian(self) -> 'np.ndarray':
        """Hessian of energy versus reduced internal coordinates

        Returns
        -------
        vspace_hessian : np.ndarray(3N - 6, 3N - 6)
        """
        if self._internal_hessian is None:
            raise NotSetError
        return np.dot(
            np.dot(self.vspace.T, self._internal_hessian), self.vspace)

    v_hessian = vspace_hessian

    def set_key_ic_number(self, number: int) -> None:
        """Set the value of key_ic_number of the system

        Arguments
        ---------
        number : int
            The value of key_ic_number going to be set
        """
        assert number >= 0
        self._k_ic_n = number
        self._reset_v_space()
        return None

    def select_key_ic(self, *indices: int) -> None:
        self._k_ic_n = 0
        indices = np.sort(np.array(indices))
        assert len(indices) <= len(self.ic)
        assert max(indices) < len(self.ic)
        for index in indices:
            self.swap_internal_coordinates(self._k_ic_n, index)
            self._k_ic_n += 1
        self._reset_v_space()
        return None

    @classmethod
    def update_to_reduced_internal(cls,
                                   internal_ob: Internal,
                                   key_ic_number: int = 0
                                   ) -> 'ReducedInternal':
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
        new_ob = deepcopy(internal_ob)
        new_ob.__class__ = cls
        new_ob.set_key_ic_number(key_ic_number)
        return new_ob

    def align_vspace(self, target: 'ReducedInternal') -> None:
        """Align vspace with a given target ReducedInternal object

        Arguments
        ---------
        target : ReducedInternal
            the target ReducedInternal object to align to
        """
        assert isinstance(target, ReducedInternal)
        overlap = np.dot(self.vspace.T, target.vspace)
        u, _, v = np.linalg.svd(overlap)
        q_min = np.dot(u, v)
        new_v = np.dot(self.vspace, q_min)
        self.set_vspace(new_v)
        return None

    def set_vspace(self, new_vspace: 'np.ndarray') -> None:
        """Set vspace of system with given values

        Arguments
        ---------
        new_vspace : np.ndarray(K, 3N - 6)
            The new value of vspace
        """
        self._vspace = new_vspace
        self._red_space = new_vspace[:, :self.key_ic_number]
        self._non_red_space = new_vspace[:, self.key_ic_number:]
        return None

    def update_to_new_structure_with_delta_v(self,
                                             delta_v: 'np.ndarray') -> None:
        """Update system to a new internal coordinates structure given a
        change in vspace delta_v

        Arguments
        ---------
        delta_v : np.ndarray(3N - 6,)
            coordinates changes in vspace
        """
        delta_v = np.array(delta_v)
        delta_ic = self._get_delta_ic_from_delta_v(delta_v)
        new_ic = delta_ic + self.ic_values
        self.set_target_ic(new_ic)
        # self.converge_to_target_ic()
        self.optimize_to_target_ic()
        self._reset_v_space()
        return None

    def _get_delta_ic_from_delta_v(self,
                                   delta_v: 'np.ndarray') -> 'np.ndarray':
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

    def _add_cc_to_ic_gradient(self, deriv, atoms):
        super(ReducedInternal, self)._add_cc_to_ic_gradient(deriv, atoms)
        self._reset_v_space()

    def _clear_ic_info(self):
        super(ReducedInternal, self)._clear_ic_info()
        self._reset_v_space()

    def _reset_v_space(self) -> None:
        """Reset vspace coordinates data, including vspace, gradient, hessian
        """
        self._red_space = None
        self._non_red_space = None
        self._vspace = None
        return None

    def _svd_of_b_matrix(self, threshold=1e-6) -> 'np.ndarray':  # tested
        """Select 3N - 6 non-singular vectors from b_matrix through SVD
        (eigenvalue) decomposition

        Returns:
        u : np.ndarray(K, 3N - 6)
            3N - 6 non-singular vectors from SVD
        """
        b_space = np.dot(self.b_matrix, self.b_matrix.T)
        values, vectors = np.linalg.eigh(b_space)
        # b_matrix shape is n * 3N
        # b_space is n * n
        basis = vectors[:, np.abs(values) > threshold]
        # for nonlinear molecules
        # TODO: need change for nonlinear molecules
        assert len(basis[0]) == self.df
        return basis

    def _reduced_unit_vectors(self) -> 'np.ndarray':  # tested
        """Generate unit vectors where every position is 0 except the
        key_ic_number position is 1

        Returns
        -------
        unit_mtx : np.ndarray(K, J), J is the number of key_ic_number
            Unit vectors with 0 everywhere except that key_ic_number position
            is 0
        """
        unit_mtx = np.zeros((len(self.ic), self.key_ic_number))
        unit_mtx[:self.key_ic_number, :self.key_ic_number] = np.eye(
            self.key_ic_number)
        return unit_mtx

    def _reduced_perturbation(self) -> 'np.ndarray':  # tested
        """Calculate the realizable purterbation in internal cooridnates from
        the unit vectors

        Returns
        -------
        purterbs : np.ndarray(K, J)
            Realizable purterbation in internal cooridnates from those unit
            vectors
        """
        unit_mtx = self._reduced_unit_vectors()
        tsfm = np.dot(self.b_matrix, pse_inv(self.b_matrix))
        return np.dot(tsfm, unit_mtx)

    def _generate_reduce_space(self, threshold=1e-6) -> None:  # tested
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
        self._red_space = v[:, np.abs(w) > threshold]
        return None

    def _nonreduce_vectors(self) -> 'np.ndarray':
        """Calculate the nonspace of reduced space

        Returns
        -------
        non_reduce_vectors : np.ndarray(K, 3N - 6 - J)
            the nonspace of reduced internal coordinates space
        """
        a_mtx = self._svd_of_b_matrix()
        rd_space = self._red_space
        prj_rd_space = np.dot(rd_space, rd_space.T)  # prj = \ket{\v} \bra{\v}
        non_reduce_vectors = a_mtx - np.dot(prj_rd_space, a_mtx)
        return non_reduce_vectors

    def _generate_nonreduce_space(self, threshold=1e-6) -> None:  # tested
        """Generate nonreduce part of vpsace matrix
        """
        d_mtx = self._nonreduce_vectors()
        w, v = diagonalize(d_mtx)
        self._non_red_space = v[:, abs(w) > threshold][:, :self.df -
                                                       len(self._red_space[0])]
        return None
