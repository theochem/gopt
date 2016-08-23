from __future__ import absolute_import, print_function
import numpy as np
from errors import NotSetError
from gaussianwrapper import GaussianWrapper


class Cartesian(object):

    def __init__(self, coordinates, numbers, charge, spin):
        self._coordinates = coordinates
        self._numbers = numbers
        self._charge = charge
        self._spin = spin
        self._energy = None
        self._energy_gradient = None
        self._energy_hessian = None

    @property
    def energy_gradient(self):  # return number array
        if self._energy_gradient == None:
            raise NotSetError(
                "The value 'energy_gradient' is None, do the calculation first")
        else:
            return self._energy_gradient

    @property
    def energy_hessian(self):  # return number array
        if self._energy_hessian == None:
            raise NotSetError(
                "The value 'energy_hessian' is None, do the calculation first")
        else:
            return self._energy_hessian

    @property
    def energy(self):
        if self._energy == None:
            raise NotSetError(
                "The value 'energy' is None, do the calculation first")
        else:
            return self._energy

    @property
    def numbers(self):
        return self._numbers

    @property
    def charge(self):
        return self._charge

    @property
    def spin(self):
        return self._spin

    @property
    def coordinates(self):
        return self._coordinates

    def energy_calculation(self, **kwargs`):
        title = kwargs.pop('title', 'untitled')
        method = kwargs.pop('method', 'g09')
        if method == "g09":
            ob = GaussianWrapper(self, title)
            self._energy, self._energy_gradient, self._energy_hessian = ob.run_gaussian_and_get_result(
                self.charge, self.spin, energy=True, gradient=True, hessian=True)

        # set self._energy
        # set self._energy_gradient
        # sel self._energy_hessian

    def distance(self, index1, index2):
        coord1 = self.coordinates[index1]
        coord2 = self.coordinates[index2]
        diff = coord1 - coord2
        distance = np.linalg.norm(diff)
        return distance

    def angle(self, index1, index2, index3):
        coord1 = self.coordinates[index1]
        coord2 = self.coordinates[index2]
        coord3 = self.coordinates[index3]
        diff_1 = coord2 - coord1
        diff_2 = coord2 - coord3
        cos_angle = np.dot(diff_1, diff_2) / \
            (np.linalg.norm(diff1) * np.linalg.norm(diff2))
        return cos_angle

# a = Cartesian(1,1)
