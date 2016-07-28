import horton as ht
import numpy as np
from saddle.TransitionSearch import *
from saddle.tstreat import TS_Treat
from saddle.optimizer.saddle_optimizer import TrialOptimizer
import pprint
import os


def test_transitionsearch_cl_h_br():
    # fn = ht.context.get_fn("../saddle/test/")
    path = os.path.dirname(os.path.realpath(__file__))
    reactant = ht.IOData.from_file(path + "/Cl_HBr.xyz")
    product = ht.IOData.from_file(path + "/Br_HCl.xyz")
    # create a object to find best fitting transition state
    ts_sample = TransitionSearch(reactant, product)
    # auto select ic for reactant and product in certain way
    ts_sample.auto_ic_select_combine()
    assert np.allclose(ts_sample.reactant.ic, np.array(
        [2.67533253, 5.56209896, 3.14159265]))
    assert np.allclose(ts_sample.product.ic, np.array(
        [5.14254763, 2.45181572, 3.14159265]))
    # auto select proper ic for ts and optimize the initial guess to as close
    # as possible
    ts_sample.auto_ts_search(opt=True, similar=ts_sample.reactant)
    ts_sample.auto_key_ic_select()  # auto select key ic for transition states
    assert abs(ts_sample._ic_key_counter - 2) < 1e-8
    ts_treat = ts_sample.create_ts_treat()

    new_ts_sample = TransitionSearch(reactant, product)
    new_ts_sample.add_bond(0, 1, "regular", [new_ts_sample.reactant, new_ts_sample.product])
    new_ts_sample.add_bond(0, 2, "regular", [new_ts_sample.reactant, new_ts_sample.product])
    new_ts_sample.add_angle(1, 0, 2, [new_ts_sample.reactant, new_ts_sample.product])
    #new_ts_sample.add_bond(1, 2, "regular", [new_ts_sample.reactant, new_ts_sample.product])
    new_ts_sample.auto_ts_search(opt=True, similar=new_ts_sample.reactant, select=False)
    new_ts_sample.auto_key_ic_select()  # auto select key ic for transition states
    ts_treat = new_ts_sample.create_ts_treat()
    ts_treat.get_v_basis()
    print ts_treat.v_matrix
    ts_treat.ts_state.get_energy_gradient_hessian(
        method="gs", title="clhbr", charge=0, spin=2)  # obtain energy, gradient, and hessian
    ts_treat.get_v_gradient()
    ts_treat.get_v_hessian()
    # print ts_treat.ts_state.energy
    # print ts_treat.v_gradient
    optimizer = TrialOptimizer(0, 2)
    optimizer.set_trust_radius_method(method="default", parameter=3)
    # optimizer._trust_radius.set_min(0.01)
    print "min, max {} {}".format(optimizer._trust_radius.max, optimizer._trust_radius.min)
    optimizer.add_a_point(ts_treat)
    optimizer.initialize_trm_for_point_with_index(0)
    result = optimizer.optimize(10)
