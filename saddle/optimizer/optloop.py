import numpy as np

from saddle.optimizer.path_point import PathPoint
from saddle.reduced_internal import ReducedInternal
from saddle.optimizer.trust_radius import trust_radius_methods
from saddle.optimizer.quasi_newton import QuasiNT
from saddle.optimizer.hessian_modify import modify_hessian_with_pos_defi
from saddle.optimizer.update_trust_radius import UpdateStep
from saddle.errors import OptError


class OptLoop:
    def __init__(self, init_structure, *_, quasi_nt, trust_rad, upd_base):
        if not isinstance(init_structure, ReducedInternal):
            raise TypeError(
                f'Improper input type {type(init_structure)} for {init_structure}'
            )
        self._point = [PathPoint(init_structure)]
        self._quasi_nt = QuasiNT(quasi_nt)
        self._trust_rad = trust_radius_methods[trust_rad]
        self._upd_base = UpdateStep(upd_base)

    def __len__(self):
        return len(self._point)

    def __getitem__(self, key):
        return self._point[key]

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

    def update_trust_radius(self, index=-1):
        target_p = self[index]
        if target_p.step:
            print(f'overwritten step for {target_p} {len(self) + index}')
        target_p.step = self._upd_base.update_step(
            old=self[index - 1], new=self[index])

    def update_hessian(self, index=-1):
        target_p = self[index]
        target_p.v_hessian = self._quasi_nt.update_hessian(
            old=self[index - 1], new=self[index])
