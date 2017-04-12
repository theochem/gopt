from __future__ import absolute_import, division, print_function

from copy import deepcopy

import numpy as np

from saddle.newopt.abclass import Point
from saddle.newopt.hessian_modifier import SaddleHessianModifier
from saddle.newopt.hessian_update import SR1
from saddle.newopt.step_scaler import TRIM
from saddle.newopt.trust_radius import DefaultTrustRadius


class Grape(object):
    def __init__(self, trust_radius, hessian_update, step_scale,
                 hessian_modifier):
        self._points = []
        self._t_r = trust_radius
        self._h_u = hessian_update
        self._s_s = step_scale
        self._h_m = hessian_modifier

    @property
    def total(self):
        return len(self._points)

    @property
    def last(self):
        try:
            return self._points[-1]
        except IndexError:
            return None

    def start_optimization(self,
                           iteration=10,
                           key_ic_number=0,
                           negative_eigen=0,
                           quasint=True,
                           init_hessian=True):
        assert self.total > 0
        assert iteration > 0
        if self.total == 1:
            if init_hessian == False:
                # if init hessian not provide, use identity
                self.last.set_hessian(np.eye(len(self.last.gradient)))
            self.modify_hessian(key_ic_number, negative_eigen)
            self.calculate_step(negative_eigen)
            self.update_to_new_point()
            self.align_last_point()
            iteration -= 1
        if self.total > 1:
            while iteration > 0:
                if quasint == True:
                    self.update_hessian()
                    self.update_hessian_with_finite_diff()
                conver_flag = self.converge_test()
                if conver_flag:
                    print("Optimization finished")
                    break
                self.modify_hessian(key_ic_number, negative_eigen)
                self.update_trust_radius(criterion="gradient")
                self.calculate_step(negative_eigen)
                self.update_to_new_point()
                self.align_last_point()
                iteration -= 1

    def add_point(self, new_point):
        assert isinstance(new_point, Point)
        copy_n_p = deepcopy(new_point)
        if self.last is None:
            self._t_r.initialize(copy_n_p)
        self._points.append(copy_n_p)
        print("this is a new point, {}".format(len(self._points)))

    def modify_hessian(self, *args, **kwargs):
        self._h_m.modify_hessian(self.last, *args, **kwargs)

    def calculate_step(self, *args, **kwargs):
        self._s_s.calculate_step(self.last, *args, **kwargs)

    def calculate_new_point(self, *args, **kwargs):
        new_point = self.last.update_point(*args, **kwargs)
        new_point.get_value()
        return new_point

    def update_trust_radius(self, *args, **kwargs):
        new_point = self.last
        pre_point = self._points[-2]
        self._t_r.update(new_point, pre_point, *args, **kwargs)

    def align_last_point(self):
        self.last.structure.align_vspace(self._points[-2].structure)
        self.last.reset_hessian()

    def _verify_new_point(self, new_point, *args, **kwargs):
        if np.linalg.norm(new_point.gradient) < np.linalg.norm(
                self.last.gradient):
            return 1
        else:
            self.last.set_trust_radius_scale(0.25)
            if self.last.trust_radius_stride < 0.1 * self._t_r.floor:
                self.last.set_trust_radius_stride(self._t_r.floor)
                return 0
            else:
                return -1

    def update_to_new_point(self, *args, **kwargs):
        new_point = self.calculate_new_point()
        verify_result = self._verify_new_point(new_point)
        # print('result', verify_result)
        while verify_result == -1:
            new_point = self.calculate_new_point()
            verify_result = self._verify_new_point(new_point)
        if verify_result == 0:
            new_point = self.calculate_new_point()
        # print("add a point with higher energy")
        self.add_point(new_point)

    def converge_test(self, g_cutoff=1e-4, *args, **kwargs):
        final_p = self.last
        pre_p = self._points[-2]
        if np.max(np.abs(final_p.structure.energy_gradient)) < g_cutoff:
            return True
        elif np.abs(final_p.value - pre_p.value) < 1e-6:
            return True
        #elif np.max(np.abs(pre_p.step)) < 3e-4:
        #    return True
        return False

    def update_hessian_with_finite_diff(self, *args, **kwargs):  # to be test
        print("running finite diff")
        new_point = self.last
        pre_point = self._points[-2]
        new_hessian = self._h_u.finite_update_hessian(pre_point, new_point)
        new_point.set_hessian(new_hessian)

    def update_hessian(self, *args, **kwargs):
        new_point = self.last
        pre_point = self._points[-2]
        new_hessian = self._h_u.update_hessian(pre_point, new_point, *args,
                                               **kwargs)
        new_point.set_hessian(new_hessian)


def basic_optimizer(number_atoms):
    hm = SaddleHessianModifier()
    ss = TRIM()
    tr = DefaultTrustRadius(number_atoms)
    hu = SR1()
    return Grape(
        trust_radius=tr, hessian_update=hu, step_scale=ss, hessian_modifier=hm)
    # def finite_diff_hessian(self):
    #     self._finite_hessian_verify(self._t_r.number_of_atoms)

    # def _finite_hessian_rows(self, number_of_atoms):
    #     point = self.last
    #     pre_point = self._points[-2]
    #     for i in range(point.stu)
    #     if (np.linalg.norm(point.gradient[:point.structure.key_ic_number]) > np.linalg.norm(point.gradient) / np.sqrt(3 * number_of_atoms - 6)) and
