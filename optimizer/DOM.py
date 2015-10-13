import numpy as np
from copy import deepcopy
from Point import Point



class DOM(object):
    @staticmethod
    def initialize(point):
        point_init = deepcopy(point)
        point_init._flag = "DOM"
        point_init._G = np.linalg.inv(point_init.second_deriv)
        point_init._stepratio = 0.5
        return point_init


    @staticmethod
    def update(point, value_func, deriv_func):
        if point._flag !="DOM":
            raise OptimizationTypeError
        n = 0
        while True:
            n += 1
            steplength = - point._stepratio * np.dot(point._G, point.first_deriv)
            new_coor = point.coordinates + steplength
            new_point = Point(new_coor)
            new_point = value_func(new_point)        
            if new_point.value < point.value:
                break
            point._stepratio *= 0.5
            if n > 20: raise FailConvergeError
        new_point = deriv_func(new_point)
        new_point._stepratio = point._stepratio
        new_point._G = np.linalg.inv(new_point.second_deriv)
        new_point._flag = "DOM"
        return new_point


    @staticmethod
    def optimize(point, value_func, deriv_func, threshold = 0.01):
        if point._flag != "DOM":
            raise OptimizationTypeError
        new_point = deepcopy(point)
        n = 0
        while np.dot(new_point.first_deriv, new_point.first_deriv) / len(new_point.coordinates) > threshold:
            n += 1
            if n > 100: raise FailConvergeError
            new_point = DOM.update(new_point, value_func, deriv_func)
        return new_point


class OptimizationTypeError(Exception):
    pass


class FailConvergeError(Exception):
    pass