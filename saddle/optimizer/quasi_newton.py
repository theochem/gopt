import numpy as np
from numpy.linalg import norm
from numpy import dot, outer, cross
from saddle.optimizer.errors import UpdateError


class QuasiNT:

    @staticmethod
    def simple_rank_one(hes, *_, sec_y, step):
        QuasiNT._verify_type(hes, sec_y, step)
        p1 = sec_y - dot(hes, step)
        numer = dot(p1, step)**2
        denor = norm(p1)**2 * norm(step)**2
        if denor == 0 or numer / denor <= 1e-18:  # in case zero division
            return hes.copy()
        update_h = hes + outer(p1, p1) / dot(p1, step)
        return update_h


    sr1 = simple_rank_one

    @staticmethod
    def powell_symmetric_broyden(hes, *_, sec_y, step):
        if np.allclose(norm(step), 0):
            raise UpdateError
        QuasiNT._verify_type(hes, sec_y, step)
        p_x = sec_y - dot(hes, step)
        p2 = (outer(p_x, step) + outer(step, p_x)) / dot(step, step)
        p3 = (dot(step, p_x) / dot(step, step)**2) * outer(step, step)
        return hes + p2 - p3


    psb = powell_symmetric_broyden

    @staticmethod
    def broyden_fletcher(hes, *_, sec_y, step):
        bind = dot(hes, step)
        p2 = outer(sec_y, sec_y) / dot(sec_y, step)
        p3 = outer(bind, bind) / dot(step, bind)
        return hes + p2 - p3


    bfgs = broyden_fletcher

    @staticmethod
    def bofill(hes, *_, sec_y, step):
        p_x = sec_y - dot(hes, step)
        numer = norm(cross(step, p_x))**2
        denor = norm(step)**2 * norm(p_x)**2
        ratio = numer / denor
        sr1_r = QuasiNT.sr1(hes, sec_y=sec_y, step=step)
        psb_r = QuasiNT.psb(hes, sec_y=sec_y, step=step)
        return (1 - ratio) * sr1_r + ratio * psb_r

    @staticmethod
    def _verify_type(old_hessian, secant, step) -> None:
        assert old_hessian.ndim == 2
        assert secant.ndim == 1
        assert step.ndim == 1

    _methods_dict = {
        'sr1': sr1,
        'psb': psb,
        'bfgs': bfgs,
        'bofill': bofill,
    }
