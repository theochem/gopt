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
            old.vspace_hessian + np.outer(delta_y, delta_y.T) / (np.dot(
                delta_y, old_struct.step))


class PSB(HessianUpdate):
    def update_hessian(self, old_struct, new_struct):
        assert isinstance(old_struct, SaddlePoint)
        assert isinstance(new_struct, SaddlePoint)
        old = old_struct.structure
        new = new_struct.structure
        y = self.secant_condition(old=old, new=new)
        delta_y = y - np.dot(old.vspace_hessian, old_struct.step)
        term2 = ((np.outer(delta_y, old_struct.step.T) +
                  np.outer(old_struct.step, delta_y.T)) /
                  np.dot(old_struct.step.T, old_struct.step))
        term3 = (np.dot(old_struct.step.T, delta_y) /
                 np.dot(old_struct.step.T, old_struct.step)**2 *
                 np.outer(old_struct.step, old_struct.step.T))
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
