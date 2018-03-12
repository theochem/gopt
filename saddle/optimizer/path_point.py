import numpy as np
from copy import deepcopy

from saddle.errors import NotSetError


class PathPoint:
    def __init__(self, red_int):
        self._instance = red_int
        self._step = None
        self._stepsize = None
        self._mod_hessian = None

    @property
    def energy(self):
        return self._instance.energy

    @property
    def x_gradient(self):
        return self._instance.energy_gradient

    @property
    def x_hessian(self):
        return self._instance.energy_hessian

    @property
    def b_matrix(self):
        return self._instance.b_matrix

    @property
    def q_gradient(self):
        return self._instance.q_gradient

    # @property
    # def q_hessian(self):
    #     return self._instance.q_hessian

    @property
    def vspace(self):
        return self._instance.vspace

    @property
    def v_gradient(self):
        return self._instance.v_gradient

    @property
    def v_hessian(self):
        if self._mod_hessian is not None:
            return self._mod_hessian
        return self.raw_hessian

    @v_hessian.setter
    def v_hessian(self, value):
        if self._mod_hessian.shape != value.shape:
            raise ValueError("The shape of input is not valid")
        if np.allclose(value, value.T):
            raise ValueError("The input Hessian is not hermitian")
        self._mod_hessian = value

    @property
    def key_ic_number(self):
        return self._instance.key_ic_number

    @property
    def df(self):
        return self._instance.df

    @property
    def raw_hessian(self):
        return self._instance.v_hessian

    @property
    def step(self):
        if self._step is not None:
            return self._step
        raise NotSetError

    @step.setter
    def step(self, value):
        if np.linalg.norm(value) > self.stepsize:
            raise ValueError
        self._step = value

    @property
    def stepsize(self):
        if self._stepsize is not None:
            return self._stepsize
        raise NotSetError

    @stepsize.setter
    def stepsize(self, value):
        assert value > 0
        self._stepsize = value

    def __repr__(self):
        return f"PathPoint object"

    def run_calculation(self, *_, method):
        self._instance.energy_calculation(method)

    def update_coordinates_with_delta_v(self, step_v):
        # this function will change the coordinates of instance
        self._instance.update_coordinates_with_delta_v(step_v)
        # initialize all the private variables
        self._step = None
        self._mod_hessian = None
        self._stepsize = None

    def copy(self):
        new_p = PathPoint(deepcopy(self._instance))
        return new_p

    def _measure_fd_hessian(self, omega=1.0, nu=1.0):
        pass
