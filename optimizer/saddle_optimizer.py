import numpy as np
import math

from copy import deepcopy
from saddle.saddlepoint import SaddlePoint


class TrialOptimizer(object):

    def __init__(self):
        self.points = []
        self._trust_radius = None

    def set_trust_radius(**kwmethod):
        method = kwmethod.pop("method", "default")
        parameter = kwmethod.pop("parameter", None)
        if parameter:
            pass
        else:
            trm_class = TRM[method]
            self._trust_radius = trm_class()
        
    def initialize_trm_for_point_with_index(self, index):
        point = self.points[index]
        self._trust_radius.initilize_point(point)

    def update_trm_for_point_with_index(self, index, method):
        point = self.points[index]
        pre_point = self.points[index - 1]
        update_method = self._trust_radius.update_trust_radius(method)
        update_method(point, pre_point)

    @property
    def latest_index(self):
        return len(self.points) - 1

    def add_a_point(self, point):
        self.points.append(point)

    def update_hessian_for_a_point(self, index, **kwmethod):
        method = kwmethod.pop("method")
        if kwmethod:
            raise TypeError('Unexpected **kwargs: {}'.format(kwmethod))
        point = self.points[index]
        if point.hessian:
            print("exits, quit updating")
        else:
            point.hessian = hessian_update_function[
                method]  # function need to be added here

    def update_hessian_for_latest_point(self, **kwmethod):
        self.update_hessian_for_a_point(self.latest_index, kwmethod)

    def tweak_hessian_for_a_point(self, index):
        point = self.points[index]
        self._tweak_hessian(point)

    def tweak_hessian_for_latest_point(self):
        point = self.points[self.latest_index]
        self._tweak_hessian(point)

    def _tweak_hessian(self, point):
        point._diagnolize_h_matrix()  # diagnolize hessian matrix
        # modify the eigenvalue of hessian to make sure there is only one
        # negative eigenvalue
        point._modify_h_matrix()
        point._reconstruct_hessian_matrix()  # reconstruct hessian matrix

    def find_stepsize_for_a_point(self, index, **kwmethod):
        point = self.points[index]
        method = kwmethod.pop("method")
        parameter = kwmethod.pop("parameter", None)
        if kwmethod:
            raise TypeError('Unexpected **kwargs: {}'.format(kwmethod))
        if method == "TRIM":
            stepsize = point._trust_region_image_potential()
        elif method == "RFO":
            pass
        point.stepsize = step_size

    # def find_stepsize_for_latest_point(self, **kwmethod):
    #     self.find_stepsize_for_a_point(self.latest_index, kwmethod)

    # def update_trust_radius_of_a_point(self, index, **kwmethod):
    #     if index == 0:
    #         raise IndexError("Cannot update trust radius method")
    #     point = self.points[index]
    #     method = kwmethod.pop('method')
    #     if kwmethod:
    #         raise TypeError('Unexpected **kwargs: {}'.format(kwmethod))
    #     trust_radius_update_method = TRM[method]
    #     trust_radius_update_method(point, pre_point)



    trm_class = {
        "default" = default_trust_radius
    }




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

    def update_trust_radius(self, method):
        if method == "energy":
            return self._energy_based_trust_radius_method
        elif method == "gradient":
            return self._gradient_based_trust_radius_method

    def _energy_based_trust_radius_method(self, point, pre_point):
        delta_m = np.dot(pre_point.g_matrix, pre_point.stepsize) + 1. / 2 * np.dot(
            np.dot(pre_point.stepsize.T, pre_point.h_matrix), pre_point.stepsize)  # approximated energy changes
        delta_u = point.value - pre_point.value  # accurate energy changes
        ratio_of_m_over_u = delta_m / delta_u
        if ratio_of_m_over_u > 2. / 3 and ratio_of_m_over_u < 3. / 2:
            point.step_control = min(max(2 * pre_point.step_control, self.min), self.max)
        elif ratio_of_m_over_u > 1. / 3 and ratio_of_m_over_u < 3.:
            point.step_control = max(pre_point.step_control, self.min)
        else:
            point.step_control = min(1. / 4 * pre_point.step_control, self.min)

    def _gradient_based_trust_radius_method(self, point, pre_point, dimensions):
        g_predict = pre_point.g_matrix + \
            np.dot(pre_point.h_matrix, pre_point.stepsize)
        norm = np.linalg.norm
        ratio_rho = (norm(g_predict) - norm(pre_point.g_matrix)) / \
            (norm(point.g_matrix) - norm(pre_point.g_matrix))
        cos_ita = np.dot((g_predict - pre_point.g_matrix), (point.g_matrix - pre_point.g_matrix)) / \
            np.dot(norm(g_predict - pre_point.g_matrix),
                   norm(point.g_matrix, pre_point.g_matrix))
        p_10 = math.sqrt(1.6424 / dimensions + 1.11 / dimensions ** 2)
        p_40 = math.sqrt(0.064175 / dimensions + 0.0946 / dimensions ** 2)
        if ratio_rho > 4. / 5 and ratio_rho < 5. / 4 and cos_ita > p_10:
            point.step_control = min(max(2 * pre_point.step_control, self.min), self.max)
        elif: ratio_rho > 1. / 5 and ratio_rho < 6. and cos_ita > p_40:
            point.step_control = max(pre_point.step, self.min)
        else:
            point.step_control = min(1. / 2 * pre_point.step, self.min)


'''EXAMPLE

a = TrialOptimizer()
a.add_a_point(point)
a.set_trust_radius(method=default)
a.initialize_trm_for_point_with_index(0)
a.update_hessian_for_latest_point(method=BFGS)
a.find_stepsize_for_a_point(method=TRIM, parameter=0.1)
new_point = point.update()
a.add_a_point(new_point)



'''
