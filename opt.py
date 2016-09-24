from __future__ import absolute_import, print_function, division
import numpy as np
from saddle.solver import ridders_solver


class GeoOptimizer(object):

    def __init__(self):
        self.points = []

    def __getitem__(self, index):
        return self.points[index]

    def converge(self, index):
        point = self.points[index]
        if max(point.gradient) <= 3e-4:
            return True
        else:
            return False

    @property
    def newest(self):
        return len(self.points) - 1

    def newton_step(self, index):
        point = self.points[index]
        return -np.dot(np.linalg.pinv(point.hessian), point.gradient)

    def add_new(self, point):
        self.points.append(point)

    def tweak_hessian(self, index, negative=0, threshold=0.005):
        point = self.points[index]
        w, v = np.linalg.eigh(point.hessian)
        negative_slice = w[: negative]
        positive_slice = w[negative:]
        negative_slice[negative_slice > -threshold] = -threshold
        positive_slice[positive_slice < threshold] = threshold
        new_hessian = np.dot(v, np.dot(np.diag(w), v.T))
        point.hessian = new_hessian

    def trust_radius_step(self, index, negative=0):
        point = self.points[index]
        c_step = self.newton_step(index)
        if np.linalg.norm(c_step) <= point.trust_radius:
            point.step = c_step
            return c_step
        w, v = np.linalg.eigh(point.hessian)
        max_w = max(w)
        # print ("special 0", np.linalg.norm(-np.dot(np.dot(v, np.dot(np.diag(1. / w), v.T)), point.gradient)) - point.trust_radius)
        def func_step(value):
            x = w.copy()
            x[:negative] = x[:negative] - value
            x[negative:] = x[negative:] + value
            new_hessian_inv = np.dot(v, np.dot(np.diag(1. / x), v.T))
            return -np.dot(new_hessian_inv, point.gradient)

        def func_value(value):
            step = func_step(value)
            return np.linalg.norm(step) - point.trust_radius

        while func_value(max_w) >= 0:
            max_w *= 2
        # print("ridder", max_w, func_value(0), np.linalg.norm(c_step) - point.trust_radius, func_value(max_w))
        result = ridders_solver(func_value, 0, max_w)
        # print ("result", result)
        step = func_step(result)
        point.step = step
        return step

    def update_trust_radius(self, index):
        point = self.points[index]
        pre_point = self.points[index - 1]
        if np.linalg.norm(point.gradient) > np.linalg.norm(pre_point.gradient):
            point.trust_radius = pre_point.trust_radius * 0.25
            return
        g_predict = pre_point.gradient + \
            np.dot(pre_point.hessian, pre_point.step)
        ratio = np.linalg.norm(g_predict) - np.linalg.norm(pre_point.gradient) / (
            np.linalg.norm(point.gradient) - np.linalg.norm(pre_point.gradient))
        if 0.8 <= ratio <= 1.25:
            point.trust_radius = pre_point.trust_radius * 2.
        elif 0.2 <= ratio <= 6:
            point.trust_radius = pre_point.trust_radius
        else:
            point.trust_radius = pre_point.trust_radius * .5
