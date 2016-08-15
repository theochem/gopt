from __future__ import absolute_import, print_function
import numpy as np
from errors import NotSetError

class Cartesian(object):
    def __init__(self, coordinates, atoms):
        self._coordinates = coordinates
        self._atoms = atoms
        self._energy = None

    def energy_gradient(self): # return number array
        pass

    def energy_hessian(self): # return number array
        pass

    @property
    def energy(self):
        if self._energy == None:
            raise NotSetError("The value is None, do the calculation to calculate it first")
        else:
            return self._energy

    def energy_calculation(self, *methods):
        pass
# a = Cartesian(1,1)
