import numpy as np
from functools import partialmethod

from saddle.errors import NotSetError, OptError
from saddle.optimizer.hessian_modify import modify_hessian_with_pos_defi
from saddle.optimizer.optloop import OptLoop
from saddle.optimizer.react_point import ReactPoint
from saddle.optimizer.quasi_newton import QuasiNT
from saddle.optimizer.step_size import Stepsize
from saddle.optimizer.trust_radius import TrustRegion
from saddle.reduced_internal import ReducedInternal


class PathLoop(OptLoop):
    def __init__(self,
                 init_structure,
                 dir_vct,
                 *_,
                 quasi_nt,
                 trust_rad,
                 upd_size,
                 method='g09',
                 max_pt=0):
        if not isinstance(init_structure, ReducedInternal):
            raise TypeError(f'Improper input type \
                {type(init_structure)} for {init_structure}')
        # TODO: possible momery saving mode
        self._point = [ReactPoint(init_structure, dir_vct)]
        self._quasi_nt = QuasiNT(quasi_nt)
        self._trust_rad = TrustRegion(trust_rad)
        self._upd_size = Stepsize(upd_size)

        # memory setting
        if max_pt == 0 or max_pt >= 2:
            self._max_pt = max_pt
        else:
            raise ValueError('max number of points is too small')

        # initialize step_size
        self._upd_size.initialize(self.new)
        self._neg = 0
        self._method = method
        self._flag = False

    @property
    def new(self):
        if len(self) < 1:
            raise OptError('Not enough points in OptLoop')
        return self[-1]

    @property
    def old(self):
        if len(self) < 2:
            raise OptError('Not enough points in OptLoop')
        return self[-2]

    def check_converge(self, cutoff=3e-4):
        if np.max(np.abs(self.new.sub_x_gradient)) < cutoff:
            return True
        return False

    path_solver = partialmethod(
        OptLoop.opt_solver, quasi_nt='bfgs', trust_rad='trim', upd_size='energy')
