from __future__ import absolute_import, print_function

from copy import deepcopy

import numpy as np

from saddle.internal import Internal
from saddle.reduced_internal import ReducedInternal


class Path_RI(ReducedInternal):

    def set_path_vector(self, vector):
        assert isinstance(vector, np.ndarray)
        self._reset_v_space()
        self._path_vector = vector

    def set_key_ic_number(self, number):
        raise NotImplementedError

    def select_key_ic(self, *indices):
        raise NotImplementedError

    def _reduced_unit_vectors(self):  # tested
        raise NotImplementedError

    @classmethod
    def update_to_reduced_internal(cls, internal_ob, key_ic_number=0):
        raise NotImplementedError

    def _reduced_perturbation(self):
        tsfm = np.dot(self.b_matrix,
                      np.linalg.pinv(self.b_matrix))
        result = np.dot(tsfm, self._path_vector.T)
        assert len(result.shape) == 1
        return result[:, None]

    @property
    def key_ic_number(self):
        return 1

    @property
    def path_vector(self):
        return self._path_vector

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
            self._vspace = deepcopy(self._non_red_space)
        return self._vspace

    @classmethod
    def update_to_path_ri(cls, internal_ob, path_vector):
        assert isinstance(internal_ob, Internal)
        new_ob = deepcopy(internal_ob)
        new_ob.__class__ = cls
        new_ob._path_vector = None
        new_ob.set_path_vector(path_vector)
        return new_ob

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
    #     b_inv = np.linalg.pinv(b)
    #     return np.dot(v.T, np.dot(b, np.dot(b_inv, change_vector)))
