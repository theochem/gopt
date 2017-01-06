from __future__ import absolute_import, division

import numpy as np

from saddle.newopt.abclass import TrustRadius, Point


class DefaultTrustRadius(TrustRadius): # need to be tested
    def __init__(self, number_of_atoms):
        self._number_of_atoms = number_of_atoms

    def update(self, target_point, pre_point, criterion):
        if criterion.lower() == "energy":
            delta_m = np.dot(pre_point.gradient, pre_point.step) + np.dot(
                np.dot(pre_point.step.T, pre_point.hessian), pre_point.step)
            delta_u = target_point.value - pre_point.value
            ratio = delta_m / delta_u
            if 2 / 3 < ratio < 3 / 2:
                value = min(max(self.floor, 2 * pre_point.trust_radius_stride), self.ceiling)
            elif 1 / 3 < ratio < 3:
                value = max(pre_point.trust_radius_stride, self.ceiling)
            else:
                value = min(1 / 4 * pre_point.trust_radius_stride, self.ceiling)
            target_point.set_trust_radius_stride(value)

    def readjust(self, point):
        assert isinstance(point, Point)
        new_stride = 1 / 4 * point.trust_radius_stride
        return new_stride

    def initialize(self, point):
        assert isinstance(point, Point)
        point.set_trust_radius_stride(self.starting)

    @property
    def number_of_atoms(self):
        return self._number_of_atoms

    @property
    def ceiling(self):
        return np.sqrt(self._number_of_atoms)

    @property
    def floor(self):
        return 0.1 * np.sqrt(self._number_of_atoms)

    @property
    def starting(self):
        return 0.35 * np.sqrt(self._number_of_atoms)
