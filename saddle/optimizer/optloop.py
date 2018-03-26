import numpy as np

from saddle.optimizer.path_point import PathPoint
from saddle.reduced_internal import ReducedInternal
from saddle.optimizer.trust_radius import TrustRegion
from saddle.optimizer.quasi_newton import QuasiNT
from saddle.optimizer.hessian_modify import modify_hessian_with_pos_defi
from saddle.optimizer.step_size import Stepsize
from saddle.errors import OptError, NotSetError


class OptLoop:
    def __init__(self, init_structure, *_, quasi_nt, trust_rad, upd_base):
        if not isinstance(init_structure, ReducedInternal):
            raise TypeError(
                f'Improper input type {type(init_structure)} for {init_structure}'
            )
        # TODO: possible momery saving mode
        self._point = [PathPoint(init_structure)]
        self._quasi_nt = QuasiNT(quasi_nt)
        self._trust_rad = TrustRegion(trust_rad)
        self._upd_base = Stepsize(upd_base)
        # initialize step_size
        self._upd_base.initialize(self.new)
        # initialize init hessian
        try:
            self.new.v_hessian
        except NotSetError:
            # set to identity
            self.new.v_hessian = np.eye(self.new.df)

    def __len__(self):
        return len(self._point)

    def __getitem__(self, key):
        return self._point[key]

    def run_calculation(self, *_, method='g09'):
        self.new.run_calculation(method=method)

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

    def update_trust_radius(self):
        target_p = self.new
        if target_p.step:
            print(f'overwritten step for {target_p}')
        target_p.step = self._upd_base.update_step(old=self.old, new=self.new)

    def update_hessian(self):
        target_p = self.new
        target_p.v_hessian = self._quasi_nt.update_hessian(
            old=self.old, new=self.new)

    def converge_test(self):
        if np.max(np.abs(self.new.x_gradient)) < 3e-4:
            return True
        return False

    def calculate_trust_step(self):
        target_p = self.new
        target_p.step = self._trust_rad.calculate_trust_step(target_p)
        # target_p.step = self._trust_rad()

    def next_step_structure(self):  # TODO: memory saving can be made later
        new_pt = self.new.copy()  # deepcopy PathPoint object
        assert isinstance(new_pt, PathPoint)
        # calculate newton step
        v_step = -np.dot(
            np.linalg.pinv(self.new.v_hessian), self.new.v_gradient)
        new_pt.update_coordinates_with_delta_v(v_step)
        return new_pt

    def finite_diff_hessian(self):
        pass

    def judge_finite_diff(self):
        pass
