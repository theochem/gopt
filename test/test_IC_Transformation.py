import numpy as np
import horton as ht
import os
from saddle import *
import saddle.optimizer as op


def test_IC_Transformation_water_add_ics():
    fn_xyz = ht.context.get_fn("test/water.xyz")
    mol = ht.IOData.from_file(fn_xyz)
    water = ICTransformation(mol)
    water.add_bond_length(0, 1)
    water.add_bond_length(1, 2)
    water.add_bend_angle(0, 1, 2)
    water.add_bond_length(1, 0)
    water.add_bend_angle(2, 1, 0)
    water.ic_differences = [1.8, 1.8]

    assert len(water.ic) == 3
    assert abs(water.ic[0] - 1.81413724) < 1e-8
    assert abs(water.ic[1] - 1.81413724) < 1e-8
    assert abs(water.ic[2] - 1.91063402) < 1e-8
    assert np.allclose(water.ic_differences, [1.8, 1.8, 0.])


def test_IC_Transformation_h2_azirine():
    fn_xyz = ht.context.get_fn("test/2h-azirine.xyz")
    mol = ht.IOData.from_file(fn_xyz)
    h2a = ICTransformation(mol)
    h2a.add_bond_length(0, 1)
    h2a.add_bond_length(1, 2)
    h2a.add_bond_length(0, 2)
    h2a.add_bond_length(2, 5)
    h2a.add_bond_length(1, 3)
    h2a.add_bond_length(1, 4)

    h2a.add_bend_angle(0, 2, 1)
    h2a.add_bend_angle(0, 1, 2)
    h2a.add_bend_angle(0, 2, 5)
    h2a.add_bend_angle(0, 1, 3)
    h2a.add_bend_angle(0, 1, 4)
    h2a.add_dihed_new(1, 0, 2, 5)
    h2a.add_dihed_new(5, 2, 0, 1)

    sample = np.array([2.87966936,2.67197956,2.35391472,2.05820441,2.03738343,2.03749261,
        1.21448386,0.87272374,2.67226332,1.99497291,1.99482514,0.04706454,0.        ])
    assert np.allclose(h2a.ic, sample)

    h2a.target_ic = [2.7, 2.5, 2.3, 2.0,1.8, 1.7,1.0, 0.7, 2.7]
    sample = np.array([2.7, 2.5, 2.3, 2.0, 1.8, 1.7, 1.0, 0.7, 2.7,1.99497291,1.99482514,0.04706454,0.        ])
    assert np.allclose(h2a.target_ic, sample)

    sample = np.array([0.17966936,0.17197956,0.05391472,0.05820441,0.23738343,0.33749261,0.21448386,
        0.17272374,-0.02773668,0.,0.,0.,0.])
    assert np.allclose(h2a.ic_differences, sample)

    h2a.add_dihed_angle(0, 1, 3, 4)
    h2a.add_dihed_angle(4, 3, 1, 0)

    sample = np.array([2.87966936,2.67197956,2.35391472,2.05820441,2.03738343,2.03749261,
        1.21448386,0.87272374,2.67226332,1.99497291,1.99482514,0.04706454,0.        ,-2.44151252])
    assert np.allclose(h2a.ic, sample)

    sample = np.array([2.7, 2.5, 2.3, 2.0, 1.8, 1.7, 1.0, 0.7, 2.7,1.99497291,1.99482514,0.04706454,0.        ,-2.44151252])
    assert np.allclose(h2a.target_ic, sample)

    sample = np.array([0.17966936,0.17197956,0.05391472,0.05820441,0.23738343,0.33749261,0.21448386,
        0.17272374,-0.02773668,0.,0.,0.,0.,0.])
    assert np.allclose(h2a.ic_differences, sample)

    h2a.target_ic = [2.7, 2.5, 2.3, 2.0,1.8, 1.7, 1.0, 0.7, 2.7, 1.8, 1.8, 0.5,0.5,-2.5]

    h2aop = h2a.generate_point_object()
    h2aop = op.DOM.initialize(h2aop)
    h2aop = op.DOM.optimize(h2aop, h2a.cost_func_value_api, h2a.cost_func_deriv_api, 0.0001)

    sample = np.array([2.65960466,2.56205311,2.28111119,2.00076777,1.80286759,1.701731,
        1.15754025,0.90348139,2.58383649,1.80227751,1.80229723,0.48997636,0.45809291,-2.49348553])
    assert np.allclose(h2a.ic, sample)


if __name__ == '__main__':
    test_IC_Transformation_h2_azirine()
    test_IC_Transformation_water_add_ics()