import numpy as np


# __all__ = ["Point"]


class Point(object):


    def __init__(self, coordinates, value = None, first_deriv = np.array([]), second_deriv = np.array([])):
        self.coordinates = coordinates
        self.value = value
        self.first_deriv = first_deriv
        self.second_deriv = second_deriv


    def __repr__(self):
        return "point(coordinates: {}, value: {}, first_deriv: {}, second_deriv: {})".format(self.coordinates.shape, self.value, self.first_deriv.shape, self.second_deriv.shape)


    def __add__(self, other):
        if not isinstance(other, Point):
            raise TypeError
        return Point(self.coordinates + other.coordinates)


    def __div__(self, other):
        if not isinstance(other, int) and not isinstance(other, float):
            raise TypeError
        return Point(self.coordinates/other)


# class NewPoint(object):

#     def __init__(self, gradient, hessian=None, initial=True, extra_parameters=None):
#         self.gradient = gradient
#         self.hessian = hessian
#         self.intial = True
#         self.extra = extra_parameters

