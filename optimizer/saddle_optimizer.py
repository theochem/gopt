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
        point.stepsize = step_size

    step_size_method = {
        "TRIM": SaddlePoint._trust_region_image_potential
        "RFO": SaddlePoint._rational_function_optimization
    }

    def find_stepsize_for_latest_point(self, **kwmethod):
        self.find_stepsize_for_a_point(self.latest_index, kwmethod)

'''EXAMPLE

a = TrialOptimizer()
a.add_a_point(point)
a.update_hessian_for_latest_point(method=BFGS)
a.find_stepsize_for_a_point(method=TRIM, parameter=0.1)

'''


