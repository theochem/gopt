import numpy as np

from saddle.optimizer.path_point import PathPoint
from saddle.reduced_internal import ReducedInternal
from saddle.optimizer.trust_radius import TrustRegion
from saddle.optimizer.quasi_newton import QuasiNT
from saddle.optimizer.hessian_modify import modify_hessian_with_pos_defi
from saddle.optimizer.step_size import Stepsize
from saddle.errors import OptError, NotSetError


class OptLoop:
    def __init__(self,
                 init_structure,
                 *_,
                 quasi_nt,
                 trust_rad,
                 upd_base,
                 neg_num=0,
                 method='g09',
                 max_pt=0):
        if not isinstance(init_structure, ReducedInternal):
            raise TypeError(
                f'Improper input type {type(init_structure)} for {init_structure}'
            )
        # TODO: possible momery saving mode
        self._point = [PathPoint(init_structure)]
        self._quasi_nt = QuasiNT(quasi_nt)
        self._trust_rad = TrustRegion(trust_rad)
        self._upd_base = Stepsize(upd_base)

        # memory setting
        if max_pt == 0 or max_pt >= 2:
            self._max_pt = max_pt
        else:
            raise ValueError('max number of points is too small')

        # initialize step_size
        self._upd_base.initialize(self.new)
        if neg_num < 0 or neg_num > self.new.key_ic_number:
            raise ValueError('# of negative eigenvalues is not valid')
        self._neg = neg_num
        self._method = method
        self._flag = False

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

    def modify_hessian(self):
        moded_hessian = modify_hessian_with_pos_defi(
            self.new.v_hessian, self._neg, key_ic=self.new.key_ic_number)
        self.new.v_hessian = moded_hessian

    def update_trust_radius(self):
        target_p = self.new
        if target_p.step:
            print(f'overwritten step for {target_p}')
        target_p.step = self._upd_base.update_step(old=self.old, new=self.new)

    def update_hessian(self):
        target_p = self.new
        target_p.v_hessian = self._quasi_nt.update_hessian(
            old=self.old, new=self.new)

    def converge_test(self, cutoff=3e-4):
        if np.max(np.abs(self.new.x_gradient)) < cutoff:
            return True
        return False

    def calculate_trust_step(self):
        step = self._trust_rad.calculate_trust_step(self.new)
        self.new.step = step
        # target_p.step = self._trust_rad()

    def next_step_structure(self):  # TODO: memory saving can be made later
        new_pt = self.new.copy()  # deepcopy PathPoint object
        assert isinstance(new_pt, PathPoint)
        # calculate newton step
        new_pt.update_coordinates_with_delta_v(new_pt.step)
        return new_pt

    def new_point_test(self, new_point):
        assert isinstance(new_point, PathPoint)
        new_point.run_calculation(method=self._method)
        if self._flag is True:
            self._flag = False
            return True
        if np.linalg.norm(new_point.x_gradient) > np.linalg.norm(
                self.new.x_gradient):
            self.new.stepsize *= 0.25
            if self.new.stepsize <= 0.1 * self._upd_base.min_s:
                self.new.stepsize = self._upd_base.min_s
                self._flag = True
            return False
        else:
            return True

    def add_new_point(self, new_point):
        assert isinstance(new_point, PathPoint)
        self._point.append(new_point)

        # 0 means unlimited store
        if self._max_pt == 0:
            return
        # delete extra points
        elif len(self) > self._max_pt:
            head = self._point.pop(0)
            del head

    def finite_diff_hessian(self, omega=1.0, nu=1.0):
        update_index = self.judge_finite_diff(omega, nu)
        for i in update_index:
            self.new.fd_hessian(self, i)

    def judge_finite_diff(self, *_, omega=1.0, nu=1.0):
        index_need_fd = []
        new = self.new
        old = self.old
        norm = np.linalg.norm
        rms = omega * norm(new.v_gradient) / np.sqrt(new.df)
        for i in range(new.key_ic_number):
            if norm(new.v_gradient[i]) > rms:
                change = new.v_hessian[:, i] - old.v_hessian[:, i]
                if norm(change) > nu * norm(old.v_hessian[:, i]):
                    index_need_fd.append(i)
        return index_need_fd
