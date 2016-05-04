import math
import numpy as np

class trust_radius(object):
    pass


class default_trust_radius(trust_radius):

    def __init__(self, num_atoms):
        self._max = math.sqrt(num_atoms)
        self._min = 1. / 10 * math.sqrt(num_atoms)
        # self._value = 0.35 * math.sqrt(num_atoms)

    def initilize_point(self, point):
        point.step_control = 0.35 * self.max

    @property
    def max(self):
        return self._max

    @property
    def min(self):
        return self._min

    # def update_trust_radius(self, method):
    #     if method == "energy":
    #         return self._energy_based_trust_radius_method
    #     elif method == "gradient":
    #         return self._gradient_based_trust_radius_method

    def _energy_based_trust_radius_method(self, point, pre_point):
        delta_m = np.dot(pre_point.v_gradient, pre_point.stepsize) + 1. / 2 * np.dot(
            np.dot(pre_point.stepsize.T, pre_point.v_hessian), pre_point.stepsize)  # approximated energy changes
        delta_u = point.ts_state.energy - pre_point.ts_state.energy  # accurate energy changes
        ratio_of_m_over_u = delta_m / delta_u
        if ratio_of_m_over_u > 2. / 3 and ratio_of_m_over_u < 3. / 2:
            print 0
            point.step_control = min(
                max(2 * pre_point.step_control, self.min), self.max)
        elif ratio_of_m_over_u > 1. / 3 and ratio_of_m_over_u < 3.:
            print 1
            point.step_control = max(pre_point.step_control, self.min)
        else:
            print 2
            point.step_control = min(1. / 4 * pre_point.step_control, self.min)

    def _gradient_based_trust_radius_method(self, point, pre_point):
        dimensions = len(point.v_gradient)
        g_predict = pre_point.v_gradient + \
            np.dot(pre_point.v_hessian, pre_point.stepsize)
        norm = np.linalg.norm
        ratio_rho = (norm(g_predict) - norm(pre_point.v_gradient)) / \
            (norm(point.v_gradient) - norm(pre_point.v_gradient))
        cos_ita = np.dot((g_predict - pre_point.v_gradient), (point.v_gradient - pre_point.v_gradient)) / \
            np.dot(norm(g_predict - pre_point.v_gradient),
                   norm(point.v_gradient - pre_point.v_gradient))
        p_10 = math.sqrt(1.6424 / dimensions + 1.11 / dimensions ** 2)
        p_40 = math.sqrt(0.064175 / dimensions + 0.0946 / dimensions ** 2)
        if 0.8 < ratio_rho < 1.25 and cos_ita > p_10:
            point.step_control = min(
                max(2 * pre_point.step_control, self.min), self.max)
            print 0
        elif 0.2 < ratio_rho < 6. and cos_ita > p_40:
            point.step_control = max(pre_point.step_control, self.min)
            print 1
        else:
            point.step_control = max(1. / 2 * pre_point.step_control, self.min) ##to be comfirmed
            print 2