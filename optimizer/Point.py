import numpy as np


# __all__ = ["Point"]


class Point(object):

    def __init__(self, coordinates, value, first_deriv, second_deriv, number_of_atoms):
        self.coordinates = coordinates
        self.value = value
        self.gradient = first_deriv
        self.hessian = second_deriv
        self.trust_radius = np.sqrt(number_of_atoms)
        self.step = None
        # self.step = None

    def __repr__(self):
        return "point(coordinates: {}, value: {}, first_deriv: {}, second_deriv: {})".format(self.coordinates.shape, self.value, self.gradient.shape, self.hessian.shape)

    def __add__(self, other):
        if not isinstance(other, Point):
            raise TypeError
        return Point(self.coordinates + other.coordinates)

    def __div__(self, other):
        if not isinstance(other, int) and not isinstance(other, float):
            raise TypeError
        return Point(self.coordinates / other)


# class NewPoint(object):

#     def __init__(self, gradient, hessian=None, initial=True, extra_parameters=None):
#         self.gradient = gradient
#         self.hessian = hessian
#         self.intial = True
#         self.extra = extra_parameters
