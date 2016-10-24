from __future__ import absolute_import, print_function

from saddle.newopt.abclass import Point

import numpy as np

class Opt(object):

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
        return self._points[-1]

    def add_point(self, new_point):
        assert isinstance(new_point, Point)
        self._points.append(new_point)

    def modify_hessian(self, *args, **kwargs):
        self._h_m.modify_hessian(self.last, *args, **kwargs)

    def calculate_step(self, *args, **kwargs):
        self._s_s.calculate_step(self.last, *args, **kwargs)

    def calculate_new_point(self, *args, **kwargs):
        pass
