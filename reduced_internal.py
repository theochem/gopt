from __future__ import absolute_import, print_function

import numpy as np

# from saddle.errors import AtomsNumberError, InputTypeError, NotSetError
from saddle.internal import Internal
from saddle.solver import diagonalize


class ReducedInternal(Internal):  # need tests

    def __init__(self, coordinates, numbers, charge, spin, key_ic_number=0):
        super(ReducedInternal, self).__init__(coordinates, numbers, charge,
                                              spin)
        self._k_ic_n = key_ic_number
        self._red_space = None
        self._nonreduce_space = None

    @property
    def df(self):
        return len(self.numbers) * 3 - 6

    @property
    def key_ic_number(self):
        return self._k_ic_n

    def vspace_transfm(self):
        pass

    def set_key_ic_number(self, number):
        self._k_ic_n = number

    @classmethod
    def update_to_reduced_internal(cls, internal_ob, key_ic_number=0):
        assert isinstance(internal_ob, Internal)
        internal_ob.__class__ = cls
        internal_ob._k_ic_n = key_ic_number

    def _svd_of_cc_to_ic_gradient(self, threshold=1e-5):
        u, s, v = np.linalg.svd(self._cc_to_ic_gradient)
        return u[:, np.abs(s) > threshold][:, :self.df]

    def _reduced_unit_vectors(self):
        unit_mtx = np.zeros((len(self.ic), self.key_ic_number))
        unit_mtx[self._k_ic_n, self._k_ic_n] = np.eye(3)
        return unit_mtx

    def _reduced_perturbation(self):
        unit_mtx = self._reduced_perturbation()
        tsfm = np.dot(self._cc_to_ic_gradient, np.linalg.pinv(
            self._cc_to_ic_gradient))
        return np.dot(tsfm, unit_mtx)

    def _generate_reduce_space(self, threshold=1e-5):
        b_mtx = self._reduced_perturbation()
        w, v = diagonalize(b_mtx)
        self._red_space = v[:, abs(w) > threshold]

    def _nonreduce_vectors(self):
        a_mtx = self._svd_of_cc_to_ic_gradient()
        rd_space = self._red_space
        prj_rd_space = np.dot(rd_space, rd_space.T)  # prj = \ket{\v} \bra{\v}
        non_reduce_vectors = a_mtx - np.dot(prj_rd_space, a_mtx)
        return non_reduce_vectors

    def _generate_nonreduce_space(self, threshold=1e-5):
        d_mtx = self._nonreduce_vectors()
        w, v = diagonalize(d_mtx)
        self._nonred_space = v[:, abs(w) > threshold][:, :self.df - len(
            self._red_space[0])]
