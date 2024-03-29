"""Reaction path internal coordinate class."""
from copy import deepcopy

import numpy as np

from saddle.internal import Internal
from saddle.reduced_internal import ReducedInternal


class PathRI(ReducedInternal):
    """Reaction path reduced internal coordinates."""

    def __init__(self, coordinates, numbers, charge, multi, path_vector, title=""):
        """Initialize reaction path instance.

        Parameters
        ----------
        coordinates : np.ndarray(N, 3)
            Cartesian Coordinates of molecules
        numbers : np.ndarray(N.)
            Atomic numbers of system
        charge : int
            Molecular charge
        multi : int
            Molecular multiplicity
        path_vector : np.ndarray(n,)
            Reaction path direction vector
        title : str, optional
            Molecule's title name

        Raises
        ------
        ValueError
            Shape of path_vactor is not a np.ndarray
        """
        super().__init__(coordinates, numbers, charge, multi, title, key_ic_number=0)
        if len(path_vector.shape) == 1:
            raise ValueError("Path vector is a 1d array")
        self._path_vector = path_vector
        self._k_ic_n = 0

    @property
    def path_vector(self):
        """np.ndarray: reaction path vector in redundant internal coordinates."""
        return self._path_vector

    @property
    def real_unit_path_vector(self):
        """np.ndarray: realizable reaction path unit vector in inernal coordinates."""
        tfm = np.dot(self.b_matrix, np.linalg.pinv(self.b_matrix))
        real_path_v = np.dot(tfm, self.path_vector)
        return real_path_v / np.linalg.norm(real_path_v)

    def set_path_vector(self, vector):
        """Set a new reaction path vector."""
        assert isinstance(vector, np.ndarray)
        self._reset_v_space()
        self._path_vector = vector

    def set_key_ic_number(self, number):
        """Not implemeted in reaction path."""
        raise NotImplementedError

    def select_key_ic(self, *indices):
        """Not implemented in reaction path."""
        raise NotImplementedError

    @classmethod
    def update_to_reduced_internal(cls, internal_ob, key_ic_number=0):
        """Not implemented in reaction path."""
        raise NotImplementedError

    def _svd_of_b_matrix(self, threshold=1e-3) -> "np.ndarray":  # tested
        # b_space is n * n
        b_space = np.dot(self.b_matrix, self.b_matrix.T)
        # b_matrix shape is n * 3N
        values, vectors = np.linalg.eigh(b_space)
        # select non singular basis set
        vectors = vectors[:, np.abs(values) > threshold]
        # project out unit path vector direction
        real_proj_mtr = np.outer(self.real_unit_path_vector, self.real_unit_path_vector)
        sub_vectors = vectors - real_proj_mtr @ vectors
        # select non singular basis set
        sub_val, sub_vec = np.linalg.eigh(sub_vectors @ sub_vectors.T)
        basis = sub_vec[:, np.abs(sub_val) > threshold]
        return basis

    # def _reduced_perturbation(self):
    #     tsfm = np.dot(self.b_matrix, pse_inv(self.b_matrix))
    #     result = np.dot(tsfm, self._path_vector.T)
    #     assert len(result.shape) == 1
    #     return result[:, None]

    # @property
    # def vspace(self):
    #     """Vspace transformation matrix from internal to reduced internal

    #     Returns
    #     -------
    #     vspace : np.ndarray(K, 3N - 6)
    #     """
    #     if self._red_space is None or self._non_red_space is None:
    #         self._generate_reduce_space()
    #         self._generate_nonreduce_space()
    #         self._vspace = self._non_red_space.copy()
    #     return self._vspace

    # TO BE determined
    @classmethod
    def update_to_path_ri(cls, internal_ob, path_vector):
        """Update a reduced internal coordinates to Path Reaction Internal."""
        assert isinstance(internal_ob, Internal)
        new_ob = deepcopy(internal_ob)
        new_ob.__class__ = cls
        new_ob._path_vector = None
        new_ob._k_ic_n = 0
        new_ob.set_path_vector(path_vector)
        return new_ob

    def set_vspace(self, new_vspace: "np.ndarray") -> None:
        """Set vspace of system with given values.

        Arguments
        ---------
        new_vspace : np.ndarray(K, 3N - 6)
            The new value of vspace
        """
        overlap = np.dot(new_vspace.T, self.path_vector)
        if np.max(np.abs(overlap)) > 1e-6:
            # project out the vector space
            no_vect_path_vspace = new_vspace - np.dot(
                np.outer(self.real_unit_path_vector, self.real_unit_path_vector),
                new_vspace,
            )
            vals, vecs = np.linalg.eigh(
                np.dot(no_vect_path_vspace, no_vect_path_vspace.T)
            )
            new_vspace = vecs[:, np.abs(vals) > 1e-5]
        self._vspace = new_vspace
        self._red_space = new_vspace[:, : self.key_ic_number]
        self._non_red_space = new_vspace[:, self.key_ic_number :]

    # @property
    # def vspace(self):
    #     """Vspace transformation matrix from internal to reduced internal
    #
    #     Returns
    #     -------
    #     vspace : np.ndarray(K, 3N - 6)
    #     """
    #     if self._vspace is None:
    #         real_vector = self._realizable_change_in_vspace(
    #             self._path_vector)
    #         real_uni_vector = real_vector / np.linalg.norm(real_vector)
    #         sub_vspace = self.pre_vspace - np.dot(
    #             np.dot(self.pre_vspace, real_uni_vector), real_uni_vector.T)
    #         threshold = 1e-6  # nonzero eigenvalues threshold
    #         w, v = diagonalize(sub_vspace)
    #         self.vspace = v[:, abs(w) > threshold]
    #         return self._vspace
    #     else:
    #         return self._vspace

    # def pre_vspace(self):
    #     """Vspace transformation matrix from internal to reduced internal
    #
    #     Returns
    #     -------
    #     vspace : np.ndarray(K, 3N - 6)
    #     """
    #     if self._red_space is None or self._non_red_space is None:
    #         self._generate_reduce_space()
    #         self._generate_nonreduce_space()
    #         self._pre_vspace = np.hstack((self._red_space,
    #                                       self._non_red_space))
    #     return self._pre_vspace

    # def _realizable_change_in_vspace(self, change_vector):
    #     v = self.vspace
    #     b = self.b_matrix
    #     b_inv = pse_inv(b)
    #     return np.dot(v.T, np.dot(b, np.dot(b_inv, change_vector)))
