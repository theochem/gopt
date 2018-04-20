from functools import partialmethod

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
                 upd_size,
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
        self._upd_size = Stepsize(upd_size)

        # memory setting
        if max_pt == 0 or max_pt >= 2:
            self._max_pt = max_pt
        else:
            raise ValueError('max number of points is too small')

        # initialize step_size
        self._upd_size.initialize(self.new)
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
        target_p.stepsize = self._upd_size.update_step(
            old=self.old, new=self.new)

    def update_hessian(self):
        target_p = self.new
        target_p.v_hessian = self._quasi_nt.update_hessian(
            old=self.old, new=self.new)

    def check_converge(self, cutoff=3e-4):
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

    def verify_new_point(self, new_point, *_, debug_fchk=None):
        assert isinstance(new_point, PathPoint)
        if debug_fchk:
            # debug choices
            new_point._instance.energy_from_fchk(debug_fchk)
        else:
            new_point.run_calculation(method=self._method)
        if self._flag is True:
            self._flag = False
            return True
        if np.linalg.norm(new_point.x_gradient) > np.linalg.norm(
                self.new.x_gradient):
            self.new.stepsize *= 0.25
            if self.new.stepsize <= 0.1 * self._upd_size.min_s:
                self.new.stepsize = self._upd_size.min_s
                self._flag = True
            return False
        else:
            return True

    def add_new_point(self, new_point):
        assert isinstance(new_point, PathPoint)

        # set new point vspace to align with old one
        new_point.instance.align_vspace(self.new.instance)

        # self.align_vspace()
        self._point.append(new_point)

        # 0 means unlimited store
        if self._max_pt == 0:
            return
        # delete extra points
        elif len(self) > self._max_pt:
            head = self._point.pop(0)
            del head

    def finite_diff_hessian(self, *_, omega=1.0, nu=1.0):
        update_index = self._judge_finite_diff(omega, nu)
        for i in update_index:
            self.new.fd_hessian(self, i)

    def _judge_finite_diff(self, omega=1.0, nu=1.0):  # not fully tested
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

    @classmethod
    def opt_solver(cls,
                   init_structure,
                   *_,
                   quasi_nt,
                   trust_rad,
                   upd_size,
                   neg_num=0,
                   method='g09',
                   max_pt=0,
                   iterations=50):
        opt = cls(
            init_structure,
            quasi_nt=quasi_nt,
            trust_rad=trust_rad,
            upd_size=upd_size,
            neg_num=neg_num,
            method=method,
            max_pt=max_pt)

        # initiate counter
        counter = 1

        # setup optimization loop
        while opt.check_converge() is False:
            if counter > 1:
                # update trust region
                opt.update_trust_radius()
                # quasi newton method for updating hessian
                opt.update_hessian()
                # regulate hessian
                opt.modify_hessian()
                # finite diff for hessian if need
                opt.finite_diff_hessian()
                # calculate new step
            opt.calculate_trust_step()
            # calculate new point
            new_point = opt.next_step_structure()
            while opt.verify_new_point(new_point) is False:
                new_point = opt.calculate_trust_step()
            # add new point to optimizer
            opt.add_new_point(new_point)

            counter += 1
            if counter > iterations:
                break

    min_solver = partialmethod(
        opt_solver, quasi_nt='bfgs', trust_rad='trim', upd_size='energy')

    ts_solver = partialmethod(
        opt_solver,
        quasi_nt='bofill',
        trust_rad='trim',
        upd_size='gradient',
        neg_num=1)
