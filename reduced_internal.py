from __future__ import absolute_import, print_function

import numpy as np

from saddle.errors import NotSetError
from saddle.internal import Internal
from saddle.solver import diagonalize


class ReducedInternal(Internal):  # need tests
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
        return len(self.numbers) * 3 - 6

    @property
    def key_ic_number(self):
        return self._k_ic_n

    @property
    def vspace(self):
        if self._red_space is None or self._non_red_space is None:
            self._generate_reduce_space()
            self._generate_nonreduce_space()
            self._vspace = np.hstack((self._red_space, self._non_red_space))
        return self._vspace


    @property
    def vspace_gradient(self):
        if self._vspace_gradient is None:
            self._vspace_gradient = np.dot(self.vspace.T, self._internal_gradient)
        return self._vspace_gradient

    @property
    def vspace_hessian(self):
        if self._vspace_hessian is None:
            self._vspace_hessian = np.dot(
                np.dot(self.vspace.T, self._internal_hessian), self.vspace)
        return self._vspace_hessian

    def set_key_ic_number(self, number):
        self._k_ic_n = number
        self._reset_v_space()

    @classmethod
    def update_to_reduced_internal(cls, internal_ob, key_ic_number=0):
        assert isinstance(internal_ob, Internal)
        internal_ob.__class__ = cls
        internal_ob._k_ic_n = key_ic_number
        internal_ob._reset_v_space()

    @classmethod
    def align_vspace(cls, one, target):
        assert isinstance(one, cls)
        assert isinstance(target, cls)
        overlap = np.dot(one.vspace.T, target.vspace)
        u, s, v = np.linalg.svd(overlap)
        q_min = np.dot(u, v)
        new_v = np.dot(one.vspace, q_min)
        one.set_vspace(new_v)

    def set_vspace(self, new_vspace):
        self._vspace = new_vspace
        self._vspace_gradient = None
        self._vspace_hessian = None

    def set_new_coordinates(self, new_coor):
        super(ReducedInternal, self).set_new_coordinates(new_coor)
        self._reset_v_space()

    def energy_from_fchk(self, abs_path, gradient=True, hessian=True):
        super(ReducedInternal, self).energy_from_fchk(abs_path, gradient, hessian)

    def energy_calculation(self, **kwargs):
        super(ReducedInternal, self).energy_calculation(**kwargs)

    def swap_internal_coordinates(self, index_1, index_2):
        super(ReducedInternal, self).swap_internal_coordinates(index_1,
                                                               index_2)
        self._reset_v_space()

    def update_to_new_structure_with_delta_v(self, delta_v):
        delta_ic = self._get_delta_ic_from_delta_v(delta_v)
        new_ic = delta_ic + self.ic_values
        self.set_target_ic(new_ic)
        self.converge_to_target_ic()
        self._reset_v_space()

    def _get_delta_ic_from_delta_v(self, delta_v):
        return np.dot(self.vspace, delta_v)

    def _add_new_internal_coordinate(self, new_ic, d, dd, atoms):  # add reset
        super(ReducedInternal, self)._add_new_internal_coordinate(new_ic, d,
                                                                  dd, atoms)
        self._reset_v_space()

    def _reset_v_space(self):
        self._red_space = None
        self._non_red_space = None
        self._vspace_gradient = None
        self._vspace_hessian = None
        self._vspace = None

    def _svd_of_cc_to_ic_gradient(self, threshold=1e-6):  # tested
        u, s, v = np.linalg.svd(self._cc_to_ic_gradient)
        return u[:, np.abs(s) > threshold][:, :self.df]

    def _reduced_unit_vectors(self):  # tested
        unit_mtx = np.zeros((len(self.ic), self.key_ic_number))
        unit_mtx[:self._k_ic_n, :self._k_ic_n] = np.eye(self._k_ic_n)
        return unit_mtx

    def _reduced_perturbation(self):  # tested
        unit_mtx = self._reduced_unit_vectors()
        tsfm = np.dot(self._cc_to_ic_gradient,
                      np.linalg.pinv(self._cc_to_ic_gradient))
        return np.dot(tsfm, unit_mtx)

    def _generate_reduce_space(self, threshold=1e-6):  # tested
        b_mtx = self._reduced_perturbation()
        w, v = diagonalize(b_mtx)
        self._red_space = v[:, abs(w) > threshold]

    def _nonreduce_vectors(self):
        a_mtx = self._svd_of_cc_to_ic_gradient()
        rd_space = self._red_space
        prj_rd_space = np.dot(rd_space, rd_space.T)  # prj = \ket{\v} \bra{\v}
        non_reduce_vectors = a_mtx - np.dot(prj_rd_space, a_mtx)
        return non_reduce_vectors

    def _generate_nonreduce_space(self, threshold=1e-6):  # tested
        d_mtx = self._nonreduce_vectors()
        w, v = diagonalize(d_mtx)
        self._non_red_space = v[:, abs(w) > threshold][:, :self.df - len(
            self._red_space[0])]
