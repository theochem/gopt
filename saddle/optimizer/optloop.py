"""Optimization loop module to run."""
from functools import partialmethod
from pathlib import Path

import numpy as np

from saddle.errors import NotSetError, OptError
from saddle.optimizer.hessian_modify import modify_hessian_with_pos_defi
from saddle.optimizer.path_point import PathPoint
from saddle.optimizer.quasi_newton import QuasiNT
from saddle.optimizer.step_size import Stepsize
from saddle.optimizer.trust_radius import TrustRegion
from saddle.reduced_internal import ReducedInternal


class OptLoop:
    """Run optimization loops."""

    def __init__(
        self,
        init_structure,
        *_,
        quasi_nt,
        trust_rad,
        upd_size,
        neg_num=0,
        method="g09",
        max_pt=0,
    ):
        """Initialize optloop instance.

        Parameters
        ----------
        init_structure : ReducedInternal
            initial guess structure of optimization
        quasi_nt : str
            method name of quasi Newton method for hessian update
        trust_rad : str
            method name of trust radius constrain method
        upd_size : str
            method name of update stepsize calculation
        neg_num : int, optional
            number of negative eigenvalues
        method : str, optional
            name of outer quantum chemistr software name
        max_pt : int, optional
            maximum points stored in memory

        Raises
        ------
        TypeError
            input initial structure is ReducedInternal
        ValueError

        """
        if not isinstance(init_structure, ReducedInternal):
            raise TypeError(
                f"Improper input type \
                {type(init_structure)} for {init_structure}"
            )
        # TODO: possible momery saving mode
        self._point = [PathPoint(init_structure)]
        self._quasi_nt = QuasiNT(quasi_nt)
        print(f"Quasi Newton Method: {self._quasi_nt.name}")
        self._trust_rad = TrustRegion(trust_rad)
        print(f"Trust radius Method: {self._trust_rad.name}")
        self._upd_size = Stepsize(upd_size)
        print(f"Stepsize control Method: {self._upd_size.name}")

        # memory setting
        if max_pt == 0 or max_pt >= 2:
            self._max_pt = max_pt
        else:
            raise ValueError("max number of points is too small")

        # initialize step_size
        self._upd_size.initialize(self.new)
        if neg_num < 0 or neg_num > self.new.key_ic_number:
            raise ValueError("# of negative eigenvalues is not valid")
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
        """int: total points in optimiztaion."""
        return len(self._point)

    def __getitem__(self, key):
        """Get nth point with dict-like index."""
        return self._point[key]

    def run_calculation(self, *_, method="g09"):
        """Run quantum chemistry calculation for all needed property for optimization.

        Parameters
        ----------
        method : str, optional
            the name of the outer quantum chemistry sofware
        """
        self.new.run_calculation(method=method)

    @property
    def new(self):
        """PathPoint: the last add point in the optimization process."""
        if len(self) < 1:
            raise OptError("Not enough points in OptLoop")
        return self[-1]

    @property
    def old(self):
        """PathPoint: the second last point in the optimization process."""
        if len(self) < 2:
            raise OptError("Not enough points in OptLoop")
        return self[-2]

    def modify_hessian(self):
        """Modify the hessian of the latest point."""
        moded_hessian = modify_hessian_with_pos_defi(
            self.new.v_hessian, self._neg, key_ic=self.new.key_ic_number
        )
        self.new.step_hessian = moded_hessian

    def update_trust_radius(self):
        """Update the stepsize of lastest point."""
        target_p = self.new
        target_p.stepsize = self._upd_size.update_step(old=self.old, new=self.new)

    def update_hessian(self):
        """Update hessian matrix of lastest point."""
        target_p = self.new
        target_p.v_hessian = self._quasi_nt.update_hessian(old=self.old, new=self.new)

    def check_converge(self, cutoff=3e-4):
        """bool: Check whether the lastest point converged."""
        if np.max(np.abs(self.new.x_gradient)) < cutoff:
            return True
        return False

    def calculate_trust_step(self):
        """Compute the update step to be equal or less than trusted stepsize."""
        step = self._trust_rad.calculate_trust_step(self.new)
        self.new.step = step
        # target_p.step = self._trust_rad()

    def next_step_structure(self):  # TODO: memory saving can be made later
        """Compute for the next structure with step."""
        new_pt = self.new.copy()  # deepcopy PathPoint object
        assert isinstance(new_pt, PathPoint)
        # calculate newton step
        new_pt.update_coordinates_with_delta_v(new_pt.step)
        return new_pt

    def verify_new_point(self, new_point, *_, debug_fchk=None):
        """bool: check the new calculated point ready to be accepted or not."""
        assert isinstance(new_point, PathPoint)
        if debug_fchk:
            # debug choices
            new_point._instance.energy_from_fchk(debug_fchk)
        else:
            new_point.run_calculation(method=self._method)
        if self._flag is True:
            self._flag = False
            return True
        if np.linalg.norm(new_point.x_gradient) > np.linalg.norm(self.new.x_gradient):
            self.new.stepsize *= 0.25
            if self.new.stepsize <= 0.1 * self._upd_size.min_s:
                self.new.stepsize = self._upd_size.min_s
                self._flag = True
            return False
        else:
            return True

    def add_new_point(self, new_point):
        """Add new point to optimization list."""
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
            self._point.pop(0)

    def finite_diff_hessian(self, *_, omega=1.0, nu=1.0):
        """Use finite difference method to compute approximate hessian."""
        update_index = self._judge_finite_diff(omega, nu)
        for i in update_index:
            self.new.fd_hessian(i)

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

    def _run_one_circle_of_calculation(self, counter):
        """Compute single iteration of optimization process."""
        if counter > 1:
            # update trust region
            self.update_trust_radius()
            # quasi newton method for updating hessian
            self.update_hessian()
            # finite diff for hessian if need
            self.finite_diff_hessian()
        # regulate hessian
        self.modify_hessian()
        # calculate new step
        self.calculate_trust_step()
        # calculate new point
        new_point = self.next_step_structure()
        while self.verify_new_point(new_point) is False:
            self.calculate_trust_step()
            new_point = self.next_step_structure()
        # add new point to optimizer
        self.add_new_point(new_point)

    @classmethod
    def opt_solver(
        cls,
        init_structure,
        *_,
        quasi_nt,
        trust_rad,
        upd_size,
        neg_num=0,
        method="g09",
        max_pt=0,
        iterations=50,
        logfile=None,
    ):
        """Create an optloop instance to optimize structure.

        Parameters
        ----------
        init_structure : ReducedInernal
            initial guess structure for optimization
        quasi_nt : str
            name of quasi Newtom update matehod
        trust_rad : str
            name of trust radius method
        upd_size : str
            name of confine update step method
        neg_num : int, optional
            number of negative eigenvalues
        method : str, optional
            name of outer quantum chemistr software
        max_pt : int, optional
            number of maximum points stored in the optimization loop
        iterations : int, optional
            number of maximum iterations of update
        logfile : Path, optional
            path to the logfile to store update process
        """
        opt = cls(
            init_structure,
            quasi_nt=quasi_nt,
            trust_rad=trust_rad,
            upd_size=upd_size,
            neg_num=neg_num,
            method=method,
            max_pt=max_pt,
        )

        # initiate counter
        counter = 1
        # print(counter)
        if logfile:
            file_path = Path(logfile)
            opt[0].instance.save_to(file_path, mode="a")
        # setup optimization loop
        while opt.check_converge() is False:
            opt._run_one_circle_of_calculation(counter)
            # print(counter)
            # if counter > 1:
            #     # update trust region
            #     opt.update_trust_radius()
            #     # quasi newton method for updating hessian
            #     opt.update_hessian()
            #     # finite diff for hessian if need
            #     opt.finite_diff_hessian()
            # # regulate hessian
            # opt.modify_hessian()
            # # calculate new step
            # opt.calculate_trust_step()
            # # calculate new point
            # new_point = opt.next_step_structure()
            # while opt.verify_new_point(new_point) is False:
            #     opt.calculate_trust_step()
            #     new_point = opt.next_step_structure()
            # # add new point to optimizer
            # opt.add_new_point(new_point)
            if logfile:
                opt[-1].instance.save_to(file_path, mode="a")
            print(counter)
            counter += 1
            if counter > iterations:
                print("Failed to converge")
                break
        print("Geometry optimization finished")
        # return opt

    min_solver = partialmethod(
        opt_solver, quasi_nt="bfgs", trust_rad="trim", upd_size="energy"
    )

    ts_solver = partialmethod(
        opt_solver, quasi_nt="bofill", trust_rad="trim", upd_size="gradient", neg_num=1
    )


class PathLoop(OptLoop):
    """Special optimization loop for optimize reaction path."""

    def check_converge(self, cutoff=1e-3):
        """bool: Check whether the lastest point converged."""
        sub_x_gradient = np.dot(
            np.dot(self.new.b_matrix.T, self.new.vspace), self.new.v_gradient
        )
        if np.max(np.abs(sub_x_gradient)) < cutoff:
            return True
        return False
