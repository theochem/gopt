from __future__ import absolute_import, print_function, division
from abclass import TrustRadius
import numpy as np


class EnergyBased(TrustRadius):

    def __init__(self, number_atoms):
        self._max = np.sqrt(number_atoms)
        self._min = 0.1 * np.sqrt(number_atoms)
        self._init = 0.35 * np.sqrt(number_atoms)

    def initialize(self, point):
        point.set_trust_radius(self._init)

    def update(self, point, new_point):
        delta_m = np.dot(point.p_gradient, point.p_step) + 0.5 * \
            np.dot(np.dot(point.p_step.T, point.p_hessian), point.p_step)
        delta_u = new_point.p_energy - point.energy
        measurement = delta_m / delta_u  # \frac{\Delta m}{\Detla U}
        if 2 / 3 < measurement < 3 / 2:
            new_trust_radius = min(
                max(2 * point.p_trust_radius, self._min), self.max)
        elif 1 / 3 < measurement < 3:
            new_trust_radius = max(point.p_trust_radius, self._min)
        else:
            new_trust_radius = max(1 / 4 * point.p_trust_radius, self._min)
        new_point.set_trust_radius(new_trust_radius)


class GradientBase(TrustRadius):

    def __init__(self, number_atoms):
        self._max = np.sqrt(number_atoms)
        self._min = 0.1 * np.sqrt(number_atoms)
        self._init = 0.35 * np.sqrt(number_atoms)
        self._dimension = 3 * number_atoms - 6

    def initialize(self, point):
        point.set_trust_radius(self._init)

    def update(self, point, new_point):
        gradient_predict = point.p_gradient + \
            np.dot(point.p_hessian, point.p_step)
        norm = np.linalg.norm  # align norm to np.linalg.norm function
        measurement_ratio = (norm(gradient_predict) - norm(point.p_gradient)) / \
            (norm(new_point.p_gradient) - norm(point.p_gradient))
        pred_part = gradient_predict - point.p_gradient
        real_part = new_point.p_gradient - point.p_gradient
        measurement_cos = np.dot(pred_part, real_part) / \
            np.dot(norm(pred_part), norm(real_part))
        p_10 = np.sqrt((1.6424 / self._dimension) +
                       1.11 / (self._dimension ** 2))
        p_40 = np.sqrt((0.064175 / self._dimension) +
                       0.0946 / (self._dimension ** 2))
        if 4 / 5 < measurement_ratio < 5 / 4 and measurement_cos > p_10:
            new_trust_radius = min(
                max(2 * point.p_trust_radius, self._min), self._max)
        elif 1 / 5 < measurement_ratio < 6 and measurement_cos > p_40:
            new_trust_radius = max(point.p_trust_radius, self._min)
        else:
            new_trust_radius = max(0.5 * point.p_trust_radius, self._min)
        new_point.set_trust_radius(new_trust_radius)
