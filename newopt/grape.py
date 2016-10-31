from __future__ import absolute_import, division, print_function

from copy import deepcopy

import numpy as np

from saddle.newopt.abclass import Point


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

    def add_point(self, new_point):
        assert isinstance(new_point, Point)
        copy_n_p = deepcopy(new_point)
        if self.last is None:
            self._t_r.initialize(copy_n_p)
        self._points.append(copy_n_p)

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

    def _verify_new_point(self, new_point, *args, **kwargs):
        if np.linalg.norm(new_point.gradient) < np.linalg.norm(
                self.last.value):
            return 1
        else:
            new_point.set_trust_radius_scale(0.25)
            if new_point < 0.1 * self._s_s.floor:
                new_point.trust_radius_scale.set_trust_radius_stride(
                    self._s_s.floor)
                return 0
            else:
                return -1

    def update_to_new_point(self, *args, **kwargs):
        new_point = self.calculate_new_point()
        verify_result = self._verify_new_point(new_point)
        # print('result', verify_result)
        while verify_result == -1:
            print('result', verify_result)
            new_point = self.calculate_new_point()
            verify_result = self._verify_new_point(new_point)
        if verify_result == 0:
            new_point = self.calculate_new_point()
        self.add_point(new_point)

    def converge_test(self, *args, **kwargs):
        final_p = self.last
        pre_p = self._points[-2]
        if np.max(np.abs(final_p.gradient)) < 5e-4:
            return True
        elif np.abs(final_p.value - pre_p.value) < 1e-6:
            return True
        elif np.max(np.abs(pre_p.step)) < 3e-4:
            return True
        return False
