import horton as ht
import numpy as np
from saddle.TransitionSearch import *
from saddle.tstreat import TS_Treat
from saddle.optimizer.saddle_optimizer import TrialOptimizer
import pprint

def test_transitionsearch_cl_h_br():
    fn = ht.context.get_fn("../saddle/test/")
    reactant = ht.IOData.from_file(fn + "Cl_HBr.xyz")
    product = ht.IOData.from_file(fn + "Br_HCl.xyz")
    ts_sample = TransitionSearch(reactant, product) # create a object to find best fitting transition state

    ts_sample.auto_ic_select_combine() #auto select ic for reactant and product in certain way
    assert np.allclose(ts_sample.reactant.ic, np.array([2.67533253, 5.56209896, 3.14159265]))
    assert np.allclose(ts_sample.product.ic, np.array([5.14254763, 2.45181572, 3.14159265]))
    ts_sample.auto_ts_search()  #auto select proper ic for ts and optimize the initial guess to as close as possible
    ts_sample.auto_key_ic_select()  #auto select key ic for transition states
    assert abs(ts_sample._ic_key_counter - 2) < 1e-8
    ts_treat = ts_sample.create_ts_treat()
    assert isinstance(ts_treat, TS_Treat)
    assert np.allclose(ts_sample.ts_state.ic, [ 3.90894008, 4.00695734, 3.14159265, 7.91589742, 0. ,0. ])
    a_matrix = ts_treat._matrix_a_eigen()
    assert np.allclose(a_matrix, np.linalg.svd(ts_treat.ts_state.b_matrix)[0][:,:4])
    b_vector = ts_treat._projection()
    ts_treat.get_v_basis()
    ortho_b = TS_Treat.gram_ortho(b_vector)
    dric = np.dot(b_vector, ortho_b)
    new_dric = [dric[:, i] / np.linalg.norm(dric[:,i]) for i in range(len(dric[0]))]
    new_dric = np.array(new_dric).T
    part1 = np.dot(new_dric, new_dric.T)
    part2 = np.dot(part1, a_matrix)
    nonredu = a_matrix - part2
    ortho_f = TS_Treat.gram_ortho(nonredu)
    rdric = np.dot(nonredu, ortho_f)
    new_rdric = [rdric[:,i] / np.linalg.norm(rdric[:, i]) for i in range(len(rdric[0]))]
    new_rdric = np.array(new_rdric).T
    test_v = np.hstack((new_dric, new_rdric))
    print test_v.shape
    assert np.allclose(ts_treat.v_matrix, test_v)
    ts_treat.ts_state.get_energy_gradient_hessian()
    ts_treat.ts_state.get_energy_gradient()
    print ts_treat.ts_state.energy, ts_treat.ts_state.gradient_matrix#, ts_treat.ts_state.hessian_matrix
    # print "x",ts_treat.ts_state.gradient_matrix
    # print "x",ts_treat.ts_state.hessian_matrix
    # print "ic",ts_treat.ts_state.ic_gradient
    # print "ic", ts_treat.ts_state.ic_hessian
    ts_treat.ts_state.gradient_ic_to_x()
    ts_treat.ts_state.hessian_ic_to_x()
    # print "x",ts_treat.ts_state.gradient_matrix
    # print "x",ts_treat.ts_state.hessian_matrix
    ts_treat.ts_state.hessian_x_to_ic()
    # print "ic", ts_treat.ts_state.ic_hessian
    ts_treat.ts_state.hessian_ic_to_x()
    ts_treat.get_v_gradient()
    ts_treat.get_v_hessian()
    print ts_treat.v_matrix.shape
    print ts_treat.v_hessian
    print "hessian", np.linalg.eigh(ts_treat.v_hessian)
    print ts_treat.v_gradient
    print ts_treat.step_control
    optimizer = TrialOptimizer()
    optimizer.set_trust_radius_method(method="default", parameter=3)
    optimizer.add_a_point(ts_treat)
    optimizer.initialize_trm_for_point_with_index(0)
    optimizer.tweak_hessian_for_latest_point()
    print np.linalg.eigh(ts_treat.v_hessian)
    optimizer.find_stepsize_for_latest_point(method="TRIM")
    print ts_treat.stepsize
    new_point = ts_treat.obtain_new_cc_with_new_delta_v(ts_treat.stepsize)
    print "new point", new_point, new_point.ts_state.coordinates
    print ts_treat.ts_state.energy
    print new_point.ts_state.energy
    print new_point.ts_state.coordinates
    another_new = optimizer.update_to_new_point_for_latest_point()
    print "another energy", another_new.ts_state.energy, another_new.ts_state.ic_gradient
    print optimizer._check_new_point_satisfied(ts_treat, another_new)
    print ts_treat.step_control
    optimizer._change_trust_radius_step(0, 0.25)
    print ts_treat.step_control
    optimizer.find_stepsize_for_latest_point(method="TRIM")
    second_new = optimizer.update_to_new_point_for_latest_point()
    print "norm",np.linalg.norm(second_new.ts_state.gradient_matrix)
    optimizer.add_a_point(second_new)
    second = optimizer._secant_condition(optimizer.points[1], optimizer.points[0])
    print "this is secand",second
    optimizer.update_hessian_for_latest_point(method="SR1")
    print optimizer.points[1].v_hessian
    print optimizer.points[0].v_hessian
    optimizer.tweak_hessian_for_latest_point()
    optimizer.find_stepsize_for_latest_point(method="TRIM")
    print "stepsize", second_new.stepsize
    third_p = ts_treat.obtain_new_cc_with_new_delta_v(ts_treat.stepsize)
    print third_p.ts_state.gradient_matrix
    # print optimizer.points[1].advanced_info['']
    # optimizer.tweak_hessian_for_latest_point()
    # optimizer.find_stepsize_for_latest_point(method="TRIM")
    # third_p = optimizer.update_to_new_point_for_latest_point()


    # print ts_treat.step_control
    # ts_treat._diagnolize_h_matrix()
    # print ts_treat.advanced_info['eigenvalues']
    # ts_treat._modify_h_matrix()
    # print ts_treat.advanced_info['eigenvalues']
    # ts_treat._reconstruct_hessian_matrix()
    # print "new hessian", ts_treat.v_hessian


    # print ts_treat.ts_state.ic_gradient
    # print "x", ts_treat.ts_state.hessian_matrix

    # start_point = ts_treat.create_a_saddle_point()

    # print test_v.shape
    # print reactant.natom


if __name__ == '__main__':
    test_transitionsearch_cl_h_br()