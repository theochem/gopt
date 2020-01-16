import numpy as np
from functools import partialmethod

from saddle.errors import OptError

# from saddle.optimizer.hessian_modify import modify_hessian_with_pos_defi
from saddle.optimizer.optloop import OptLoop
from saddle.optimizer.react_point import ReactPoint
from saddle.optimizer.quasi_newton import QuasiNT
from saddle.optimizer.step_size import Stepsize
from saddle.optimizer.trust_radius import TrustRegion
from saddle.reduced_internal import ReducedInternal


class PathLoop(OptLoop):
    def __init__(
        self,
        init_structure,
        dir_vct,
        *_,
        quasi_nt,
        trust_rad,
        upd_size,
        method="g09",
        max_pt=0,
    ):
        if not isinstance(init_structure, ReducedInternal):
            raise TypeError(
                f"Improper input type \
                {type(init_structure)} for {init_structure}"
            )
        # TODO: possible momery saving mode
        self._point = [ReactPoint(init_structure, dir_vct)]
        self._quasi_nt = QuasiNT(quasi_nt)
        self._trust_rad = TrustRegion(trust_rad)
        self._upd_size = Stepsize(upd_size)

        # memory setting
        if max_pt == 0 or max_pt >= 2:
            self._max_pt = max_pt
        else:
            raise ValueError("max number of points is too small")

        # initialize step_size
        self._upd_size.initialize(self.new)
        self._neg = 0
        self._method = method
        self._flag = False

    @property
    def new(self):
        if len(self) < 1:
            raise OptError("Not enough points in OptLoop")
        return self[-1]

    @property
    def old(self):
        if len(self) < 2:
            raise OptError("Not enough points in OptLoop")
        return self[-2]

    def check_converge(self, cutoff=3e-4):
        if np.max(np.abs(self.new.sub_x_gradient)) < cutoff:
            return True
        return False

    def verify_new_point(self, new_point, *_, debug_fchk=None):
        assert isinstance(new_point, ReactPoint)
        if debug_fchk:
            # debug choices
            new_point._instance.energy_from_fchk(debug_fchk)
        else:
            new_point.run_calculation(method=self._method)
        if self._flag is True:
            self._flag = False
            return True
        if np.linalg.norm(new_point.sub_x_gradient) > np.linalg.norm(
            self.new.sub_x_gradient
        ):
            self.new.stepsize *= 0.25
            if self.new.stepsize <= 0.1 * self._upd_size.min_s:
                self.new.stepsize = self._upd_size.min_s
                self._flag = True
            return False
        else:
            return True

    @classmethod
    def opt_solver(
        cls,
        init_structure,
        dir_vect,
        *_,
        quasi_nt,
        trust_rad,
        upd_size,
        method="g09",
        max_pt=0,
        iterations=50,
    ):
        opt = cls(
            init_structure,
            dir_vect,
            quasi_nt=quasi_nt,
            trust_rad=trust_rad,
            upd_size=upd_size,
            method=method,
            max_pt=max_pt,
        )

        # initiate counter, neg_num
        counter = 1

        # setup optimization loop
        while opt.check_converge() is False:
            print(counter)
            if counter > 1:
                # update trust region
                opt.update_trust_radius()
                # quasi newton method for updating hessian
                opt.update_hessian()
                # finite diff for hessian if need
                # opt.finite_diff_hessian()
            # regulate hessian
            opt.modify_hessian()
            # calculate new step
            opt.calculate_trust_step()
            # calculate new point
            new_point = opt.next_step_structure()
            while opt.verify_new_point(new_point) is False:
                opt.calculate_trust_step()
                new_point = opt.next_step_structure()
            # add new point to optimizer
            opt.add_new_point(new_point)

            counter += 1
            if counter > iterations:
                print("Failed to converge")
                break
        print("Geometry optimization finished")

    path_solver = partialmethod(
        opt_solver, quasi_nt="bfgs", trust_rad="trim", upd_size="energy"
    )
