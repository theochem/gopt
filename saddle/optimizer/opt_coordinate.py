import numpy as np
from saddle.errors import NotSetError


class OptCoordinate:
    def __init__(self, reduce_ic):
        self._reduce_ic = reduce_ic
        self._step = None
        self._stepsize = None
        self._v_hessian = None

    @property
    def energy(self):
        return self._reduce_ic.energy

    @property
    def x_gradient(self):
        return self._reduce_ic.x_gradient

    @property
    def q_gradient(self):
        return self._reduce_ic.q_gradient

    @property
    def v_gradient(self):
        return self._reduce_ic.v_gradient

    @property
    def dof(self):
        return self._reduce_ic.natom - 6

    @property
    def v_hessian(self):
        if self._v_hessian:
            return self._v_hessian
        raise NotSetError

    @property
    def step(self):
        if self._step:
            return self._step
        raise NotSetError

    @property
    def stepsize(self):
        if self._stepsize:
            return self._stepsize
        raise NotSetError

    @property
    def b_matrix(self):
        return self._reduce_ic.b_matrix

    def set_v_hessian(self, new_hessian):
        self._v_hessian = new_hessian

    def set_new_step(self, new_step):
        self._step = new_step

    def set_new_stepsize(self, new_stepsize):
        self._stepsize = new_stepsize
