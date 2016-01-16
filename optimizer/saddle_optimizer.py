import numpy as np

from copy import deepcopy
from saddle.saddlepoint import SaddlePoint

class TrialOptimizer(object):

    def __init__(self):
        self.points = []

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
            print ("exits, quit updating")
        else:
            point.hessian = hessian_update_function[method] # function need to be added here

    def update_hessian_for_latest_point(self, **kwmethod):
        self.update_hessian_for_a_point(self.latest_index, kwmethod)

    def tweak_hessian_for_a_point(self, index):
        point = self.points[index]
        self._tweak_hessian(point)

    def tweak_hessian_for_latest_point(self):
        point = self.points[self.latest_index]
        self._tweak_hessian(point)

    def _tweak_hessian(self, point):
        point._diagnolize_h_matrix() # diagnolize hessian matrix
        point._modify_h_matrix() # modify the eigenvalue of hessian to make sure there is only one negative eigenvalue
        point._reconstruct_hessian_matrix() # reconstruct hessian matrix

    def find_stepsize_for_a_point(self, index, **kwmethod):
        point = self.points[index]
        method = kwmethod.pop("method")
        parameter = kwmethod.pop("parameter", None)
        if kwmethod:
            raise TypeError('Unexpected **kwargs: {}'.format(kwmethod))
        function_to_calculate_step = TrialOptimizer.step_size_method[method]
        if parameter:
            step_size = function_to_calculate_step(point, parameter)
        else:
            step_size = function_to_calculate_step(point)
        stepsize = max(min(self.stepsize, self.step_control.max_s), self.step_control.min_s)
        point.stepsize = step_size

    def find_stepsize_for_latest_point(self, **kwmethod):
        self.find_stepsize_for_a_point(self.latest_index, kwmethod)

    def update_trust_radius_of_a_point(self, index, **kwmethod):
        if index == 0:
            raise IndexError("Cannot update trust radius method")
        point = self.points[index]
        method = kwmethod.pop('method')
        if kwmethod:
            raise TypeError('Unexpected **kwargs: {}'.format(kwmethod))
        trust_radius_update_method = TRM[method]
        trust_radius_update_method(point, pre_point)

    _TRM = { # trust radius update method
        "Energy-based": 
        "Gradient-based": TrialOptimizer._gradient_based_trm
    }

    @staticmethod
    def _gradient_based_trm(point, pre_point):
        g_predict = pre_point.g_matrix + np.dot(pre_point.h_matrix, pre_point.stepsize)
        norm = np.linalg.norm
        ratio_rho = (norm(g_predict) - norm(pre_point.g_matrix)) / (norm(point.g_matrix) - norm(pre_point.g_matrix))



'''EXAMPLE

a = TrialOptimizer()
a.add_a_point(point)
a.update_hessian_for_latest_point(method=BFGS)
a.find_stepsize_for_a_point(method=TRIM, parameter=0.1)


'''


