import numpy as np

from numpy.testing import assert_raises
from unittest import TestCase
from copy import deepcopy

from pkg_resources import Requirement, resource_filename

from saddle.optimizer.optloop import OptLoop
from saddle.reduced_internal import ReducedInternal
from saddle.optimizer.path_point import PathPoint
from saddle.optimizer.trust_radius import TrustRegion
from saddle.optimizer.quasi_newton import QuasiNT
from saddle.optimizer.hessian_modify import modify_hessian_with_pos_defi
from saddle.optimizer.step_size import Stepsize
from saddle.iodata import IOData


class TestOptLoop(TestCase):
    def setUp(self):
        mol_path = resource_filename(
            Requirement.parse('saddle'), 'data/water.xyz')
        mol = IOData.from_file(mol_path)
        red_int = ReducedInternal(mol.coordinates, mol.numbers, 0, 1)
        red_int.add_bond(0, 1)
        red_int.add_bond(1, 2)
        red_int.add_angle_cos(0, 1, 2)
        self.mol = red_int

    def setup_opt(self):
        self.mol.select_key_ic(0)
        fchk_file = resource_filename(
            Requirement.parse('saddle'), 'data/water_old.fchk')
        self.mol.energy_from_fchk(fchk_file)
        opt = OptLoop(
            self.mol, quasi_nt='bfgs', trust_rad='trim', upd_size='energy')
        return opt

    def test_first_step(self):
        opt = self.setup_opt()
        opt.calculate_trust_step()
        # ref step
        r_p = opt.new
        ref_step = TrustRegion.trim(r_p.v_hessian, r_p.v_gradient,
                                    r_p.stepsize)
        assert np.allclose(ref_step, opt.new.step)

    def test_new_struct(self):
        opt = self.setup_opt()
        opt.calculate_trust_step()
        opt_new_p = opt.next_step_structure()
        # ref new structure
        ref_p = deepcopy(self.mol)
        ref_p.update_to_new_structure_with_delta_v(opt[0].step)

        assert np.allclose(opt_new_p.instance.coordinates, ref_p.coordinates)

    def test_verify_new_point(self):
        opt = self.setup_opt()
        opt.calculate_trust_step()

        # generate new point
        new_p = opt.next_step_structure()
        fchk_file = resource_filename(
            Requirement.parse('saddle'), 'data/new_step_water.fchk')
        result = opt.verify_new_point(new_p, debug_fchk=fchk_file)
        assert result is True
        opt.add_new_point(new_p)
        assert len(opt) == 2
        assert np.allclose(opt.new.vspace, opt.old.vspace)
        assert np.allclose(opt.new.key_ic_number, opt.new.key_ic_number)

    def test_new_opt_loop(self):
        # bfgs update
        opt = self.setup_opt()
        opt.calculate_trust_step()

        new_p = opt.next_step_structure()
        fchk_file = resource_filename(
            Requirement.parse('saddle'), 'data/new_step_water.fchk')
        result = opt.verify_new_point(new_p, debug_fchk=fchk_file)
        assert result is True
        opt.add_new_point(new_p)
        opt.update_trust_radius()
        # energy criterion update
        s = opt.old.step
        g = opt.old.v_gradient
        h = opt.old.v_hessian
        pred_e_diff = np.dot(g, s) + 0.5 * np.dot(np.dot(s.T, h), s)
        real_e_diff = opt.new.energy - opt.old.energy
        assert np.allclose((pred_e_diff / real_e_diff), 1.01485, atol=1e-4)
        assert opt.new.stepsize == 2 * opt.old.stepsize

    def test_opt_initialize(self):
        opt = OptLoop(
            self.mol, quasi_nt='bfgs', trust_rad='trim', upd_size='energy')
        assert len(opt) == 1
        assert opt._max_pt == 0
        assert opt._neg == 0
        assert opt._upd_size._max_s == np.sqrt(3)
        assert opt[0].stepsize == 0.35 * np.sqrt(3)
        assert np.allclose(opt[0].v_hessian, np.eye(3))

        opt = OptLoop(
            self.mol,
            quasi_nt='bfgs',
            trust_rad='trim',
            upd_size='energy',
            max_pt=2)
        assert opt._max_pt == 2

        with assert_raises(ValueError):
            opt = OptLoop(
                self.mol,
                quasi_nt='bfgs',
                trust_rad='trim',
                upd_size='energy',
                max_pt=1)

        with assert_raises(ValueError):
            opt = OptLoop(
                self.mol,
                quasi_nt='bfgs',
                trust_rad='trim',
                upd_size='energy',
                neg_num=1)
