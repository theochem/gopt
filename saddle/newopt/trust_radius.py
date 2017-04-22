from __future__ import absolute_import, division

import numpy as np

from .abclass import Point, TrustRadius

__all__ = ('DefaultTrustRadius', )


class DefaultTrustRadius(TrustRadius):  # need to be tested
    def __init__(self, number_of_atoms, criterion='energy'):
        self._number_of_atoms = number_of_atoms
        self._criterion = criterion

    def update(self, target_point, pre_point):
        if self._criterion == "energy":
            delta_m = np.dot(pre_point.gradient, pre_point.step) + np.dot(
                np.dot(pre_point.step.T, pre_point.hessian), pre_point.step)
            delta_u = target_point.value - pre_point.value
            ratio = delta_m / delta_u
            if 2 / 3 < ratio < 3 / 2:
                value = min(
                    max(self.floor, 2 * pre_point.trust_radius_stride),
                    self.ceiling)
            elif 1 / 3 < ratio < 3:
                value = max(pre_point.trust_radius_stride, self.ceiling)
            else:
                value = min(1 / 4 * pre_point.trust_radius_stride,
                            self.ceiling)
            target_point.set_trust_radius_stride(value)
        elif self._criterion == 'gradient':

            def p_10(d):
                return np.sqrt(1.6424 / d + 1.11 / d**2)

            def p_40(d):
                return np.sqrt(0.064175 / d + 0.0946 / d**2)

            g_pre = pre_point.gradient + np.dot(pre_point.hessian,
                                                pre_point.step)
            norm = np.linalg.norm
            rho = ((norm(g_pre) - norm(pre_point.gradient)) /
                   (norm(target_point.gradient) - norm(pre_point.gradient)))
            cos_theta = np.dot(
                g_pre - pre_point.gradient,
                target_point.gradient - pre_point.gradient) / np.dot(
                    norm(g_pre - pre_point.gradient),
                    norm(target_point.gradient - pre_point.gradient))
            if (0.8 < rho < 1.25 and
                    p_10(3 * self._number_of_atoms - 6) < cos_theta):
                print('trust 1')
                value = min(
                    max(self.floor, 2 * pre_point.trust_radius_stride),
                    self.ceiling)
            elif (0.2 < rho < 6 and
                  p_40(3 * self._number_of_atoms - 6) < cos_theta):
                print('trust 2')
                value = max(pre_point.trust_radius_stride, self.floor)
            else:
                print('trust 3')
                value = min(0.5 * pre_point.trust_radius_stride, self.floor)
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
