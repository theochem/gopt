from __future__ import absolute_import, print_function, division

from saddle.newopt.abclass import Point

from copy import deepcopy

import numpy as np

class Grape(object):

    def __init__(self, trust_radius, hessian_update, step_scale, hessian_modifier):
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
        try: return self._points[-1]
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
        return new_point

    def verify_new_point(self, new_point, *args, **kwargs):
        if new_point.value < self.last.value:
            return 1
        else:
            new_point.set_trust_radius_scale(0.25)
            if new_point < 0.1 * self._s_s.floor:
                new_point.set_trust_radius_stride(self._s_s.floor)
                return 0
            else:
                return -1
