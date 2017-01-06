from __future__ import absolute_import, division, print_function

import numpy as np

from saddle.newopt.saddle_point import SaddlePoint
from saddle.reduced_internal import ReducedInternal


class HessianUpdate(object):
    def secant_condition(self, old, new):
        assert isinstance(old, ReducedInternal)
        assert isinstance(new, ReducedInternal)
        delta_g = (new.vspace_gradient - old.vspace_gradient)
        delta_v = np.dot(
            np.dot(new.b_matrix.T, (new.vspace - old.vspace)),
            new.vspace_gradient)
        delta_b = np.dot((new.b_matrix - old.b_matrix).T,
                         new.internal_gradient)
        y = delta_g - np.dot(
            np.dot(new.vspace.T, np.linalg.pinv(new.b_matrix).T),
            (delta_v + delta_b))
        return y

    def finite_update_hessian(self, old_struct, new_struct, epsilon=0.001):
        update_index = self._update_index(old_struct, new_struct)
        for i in update_index:
            tmp_red_int = deepcopy(new_struct)
            delta_v = np.zeros(tmp_red_int.structure.df.shape[1], float)
            delta_v[i] = 1  # create a unit vector that is zero except i
            tmp_red_int.structure.update_to_new_structure_with_delta_v(delta_v *
                                                                    epsilon)
            tmp_red_int.structure.align_vspace(new_struct.structure)
            tmp_red_int.structure.nergy_calculation()
            part1 = (tmp_red_int.gradient - new_struct.gradient) / epsilon
            part2 = np.dot(new_struct.ure.vspace.T,
                           np.linalg.pinv(new_struct.ure.b_matrix.T))
            part3 = np.dot(
                np.dot(new_struct.ure.b_matrix.T,
                       (tmp_red_int.structure.vspace - new_struct.ure.vspace) /
                       epsilon), new_struct.gradient)
            part4 = np.dot(
                (tmp_red_int.structure.b_matrix - new_struct.ure.b_matrix).T /
                epsilon, new_struct.internal_gradient)
            h_vector = part1 - np.dot(part2, part3 + part4)
            new_struct._hessian[i, :] = h_vector
            new_struct._hessian[:, i] = h_vector

    def _update_index(self, old_struct, new_struct):
        # update_index = self._finite_reduce(self)  # obtain reduced ic need fd.
        update_index = []
        for i in range(new_struct.structure.key_ic_number):
            condition1 = (np.linalg.norm(new_struct.gradient[i]) >
                          np.linalg.norm(new_struct.gradient) /
                          np.sqrt(new_struct.structure.df))
            unit_vector = np.zeros(new_struct.structure.vspace.shape[1], foat)
            unit_vector[i] = 1
            condition2 = (
                np.linalg.norm(
                    np.dot(new_struct.hessian, unit_vector) - np.dot(
                        old_struct.hessian, unit_vector)) > 1. *
                np.linalg.norm(np.dot(old_struct.hessian, unit_vector)))
            if condition1 and condition2:
                update_index.append(i)
        return update_index

class SR1(HessianUpdate):
    def update_hessian(self, old_struct, new_struct):
        assert isinstance(old_struct, SaddlePoint)
        assert isinstance(new_struct, SaddlePoint)
        old = old_struct.structure
        new = new_struct.structure
        y = self.secant_condition(old=old, new=new)
        delta_y = y - np.dot(old.vspace_hessian, old_struct.step)
        numerator = np.dot(delta_y, old_struct.step)**2
        denominator = np.sum(delta_y**2) * np.sum(old_struct.step**2)
        if numerator / denominator <= 1e-18:
            return old.vspace_hessian.copy()
        else:
            return old.vspace_hessian + np.outer(delta_y, delta_y.T) / (np.dot(
                delta_y, old_struct.step))


class PSB(HessianUpdate):
    def update_hessian(self, old_struct, new_struct):
        assert isinstance(old_struct, SaddlePoint)
        assert isinstance(new_struct, SaddlePoint)
        old = old_struct.structure
        new = new_struct.structure
        y = self.secant_condition(old=old, new=new)
        delta_y = y - np.dot(old.vspace_hessian, old_struct.step)
        term2 = ((np.outer(delta_y, old_struct.step.T) + np.outer(
            old_struct.step, delta_y.T)) /
                 np.dot(old_struct.step.T, old_struct.step))
        term3 = (np.dot(old_struct.step.T, delta_y) /
                 np.dot(old_struct.step.T, old_struct.step)
                 **2 * np.outer(old_struct.step, old_struct.step.T))
        return old.vspace_hessian + term2 - term3


class BFGS(HessianUpdate):
    def update_hessian(self, old_struct, new_struct):
        assert isinstance(old_struct, SaddlePoint)
        assert isinstance(new_struct, SaddlePoint)
        old = old_struct.structure
        new = new_struct.structure
        y = self.secant_condition(old=old, new=new)
        delta_y = y - np.dot(old.vspace_hessian, old_struct.step)
        H_s = np.dot(old.vspace_hessian, old_struct.step)
        term2 = np.outer(y, y.T) / np.dot(y, old_struct.step)
        term3 = np.outer(H_s, H_s.T) / np.dot(old_struct.step, H_s)
        return old.vspace_hessian + term2 - term3
