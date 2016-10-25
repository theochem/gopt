from __future__ import absolute_import, print_function

import numpy as np

from saddle.errors import NotSetError
from saddle.newopt.abclass import Point
from saddle.reduced_internal import ReducedInternal


class SaddlePoint(Point):
    def __init__(self, structure):
        assert isinstance(structure, ReducedInternal)
        self._structure = structure
        self._step = None
        self._trust_radius_stride = None
        if self._structure.vspace_hessian is None:
            raise NotSetError
        self._hessian = self._structure.vspace_hessian.copy()

    @property
    def gradient(self):
        if self._structure.vspace_gradient is None:
            raise NotSetError
        return self._structure.vspace_gradient

    @property
    def hessian(self):
        return self._hessian

    @property
    def step(self):
        if self._step is None:
            raise NotSetError
        return self._step

    @property
    def trust_radius_stride(self):
        if self._trust_radius_stride is None:
            raise NotSetError
        return self._trust_radius_stride

    @property
    def value(self):
        if self._structure.energy is None:
            raise NotSetError
        return self._structure.energy

    def set_step(self, step):
        self._step = step

    def set_trust_radius_stride(self, stride):
        self._trust_radius_stride = stride

    def set_hessian(self, hessian):
        self._hessian = hessian
