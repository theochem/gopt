from __future__ import absolute_import, division

import numpy as np

from saddle.opt.abclass import TrustRadius


class DefaultTrustRadius(TrustRadius):
    def __init__(self, number_of_atoms):
        self._number_of_atoms = number_of_atoms

    def update(self, target_point, pre_point, criterion):
        if criterion.lower() == "energy":
            delta_m = np.dot(pre_point.gradient, pre_point.step) + np.dot(
                np.dot(pre_point.step.T, pre_point.hessian), pre_point.step)
            delta_u = target_point.energy - pre_point.energy
            ratio = delta_m / delta_u
            if 2 / 3 < ratio < 3 / 2:
                value = min(max(self.floor, 2 * pre_point.step), self.ceiling)
            elif 1 / 3 < ratio < 3:
                value = max(pre_point.step, self.ceiling)
            else:
                value = min(1 / 4 * pre_point.step, self.ceiling)
            target_point.set_trust_radius_stride(value)

    def readjust(self, point, target_point):
        new_stride = 1 / 4 * point.trust_radius_stride
        return new_stride

    @property
    def ceiling(self):
        return np.sqrt(self._number_of_atoms)

    @property
    def floor(self):
        return 0.1 * np.sqrt(self._number_of_atoms)

    @property
    def starting(self):
        return 0.35 * np.sqrt(self._number_of_atoms)
