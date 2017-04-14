from __future__ import absolute_import, print_function

from copy import deepcopy

import numpy as np

from ..errors import NotSetError
from ..reduced_internal import ReducedInternal
from .abclass import Point


class SaddlePoint(Point):
    def __init__(self, structure):
        assert isinstance(structure, ReducedInternal)
        self._structure = deepcopy(structure)
        self._step = None
        self._trust_radius_stride = None
        if self._structure.vspace_hessian is None:
            raise NotSetError
        self._hessian = self._structure.vspace_hessian.copy()

    @property
    def structure(self):
        return self._structure

    @property
    def gradient(self):
        if self._structure.vspace_gradient is None:
            raise NotSetError
        return self._structure.vspace_gradient

    @property
    def hessian(self):
        if self._hessian is None:
            self._hessian = self._structure.vspace_hessian.copy()
        return self._hessian

    @property
    def step(self):
        if self._step is None:
            raise NotSetError
        return self._step

    @property
    def trust_radius_stride(self):
        if self._trust_radius_stride is None:
            raise NotSetError
        return self._trust_radius_stride

    @property
    def value(self):
        if self._structure.energy is None:
            raise NotSetError
        return self._structure.energy

    def get_value(self):
        self._structure.energy_calculation()

    def set_step(self, step):
        self._step = step

    def set_trust_radius_stride(self, stride):
        self._trust_radius_stride = stride

    def set_trust_radius_scale(self, scale=1.0):
        assert scale > 0.
        self._trust_radius_stride *= scale

    def set_hessian(self, hessian):
        self._hessian = hessian

    def reset_hessian(self):
        self._hessian = None

    def update_point(self):  # to be tested
        if self.step is None:
            raise NotSetError
        new_self = deepcopy(self)
        new_self._structure.update_to_new_structure_with_delta_v(new_self.step)
        new_self._reset_saddle_point()
        return new_self

    # def finite_diff_hessian(self, epsilon=0.001):
    #     tmp_self = deepcopy(self)
    #     for i in range(key_ic_number):
    #         delta_v = np.zeros(tmp_self.structure.df.shape[1], float)
    #         delta_v[i] = 1  # create a unit vector that is zero except i
    #         tmp_self.structure.update_to_new_structure_with_delta_v(delta_v *
    #                                                                 epsilon)
    #         tmp_self.align_vspace(self)
    #         tmp_self.energy_calculation()
    #         part1 = (tmp_self.gradient - self.gradient) / epsilon
    #         part2 = np.dot(self.structure.vspace.T,
    #                        np.linalg.pinv(self.structure.b_matrix.T))
    #         part3 = np.dot(
    #             np.dot(self.structure.b_matrix.T,
    #                    (tmp_self.structure.vspace - self.structure.vspace) /
    #                    epsilon), self.gradient)
    #         part4 = np.dot(
    #             (tmp_self.structure.b_matrix - self.structure.b_matrix).T /
    #             epsilon, self.internal_gradient)
    #         h_vector = part1 - np.dot(part2, part3 + part4)
    #         self._hessian[i, :] = h_vector
    #         self._hessian[:, i] = h_vector

    def _reset_saddle_point(self):
        self._step = None
        self._trust_radius_stride = None
        self._hessian = None
