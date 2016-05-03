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
    ts_treat.get_v_basis() # obtain V basis for ts_treat
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
    assert np.allclose(ts_treat.v_matrix, test_v)

    ts_treat.ts_state.get_energy_gradient_hessian() #obtain energy, gradient, and hessian
    ts_treat.ts_state.get_energy_gradient()
    print "first obtain energy and gradient", ts_treat.ts_state.energy, ts_treat.ts_state.gradient_matrix#, ts_treat.ts_state.hessian_matrix
    ts_treat.get_v_gradient()
    ts_treat.get_v_hessian()
    print ts_treat.v_hessian.shape
    print "hessian", np.linalg.eigh(ts_treat.v_hessian)
    print ts_treat.v_gradient
    print ts_treat.step_control
    print "start optimization-------------------"
    optimizer = TrialOptimizer()
    optimizer.set_trust_radius_method(method="default", parameter=3)
    optimizer.add_a_point(ts_treat)
    optimizer.initialize_trm_for_point_with_index(0)
    optimizer.tweak_hessian_for_latest_point()
    # print np.linalg.eigh(ts_treat.v_hessian)
    optimizer.find_stepsize_for_latest_point(method="TRIM")
    print "first point step control",ts_treat.step_control
    second_point = optimizer.update_to_new_point_for_latest_point()
    print "second point information", second_point.step_control
    print "satisfied check",optimizer._check_new_point_converge(ts_treat, second_point) # something need to be fixed
    print optimizer.veryfy_new_point_with_index(0, second_point)
    print ts_treat.step_control
    new_second_point = optimizer.update_to_new_point_for_latest_point()
    optimizer.add_a_point(new_second_point)
    optimizer.verify_convergence_for_latest_point()
    # need to update trm. method need to be implelemented here

    # # # new_point = ts_treat.obtain_new_cc_with_new_delta_v(ts_treat.stepsize)
    # # print "new point", new_point, new_point.ts_state.coordinates
    # print ts_treat.ts_state.energy
    # # print new_point.ts_state.energy
    # # print new_point.ts_state.coordinates

    # print "another energy", another_new.ts_state.energy, another_new.ts_state.ic_gradient
    # print "satisfied check",optimizer._check_new_point_converge(ts_treat, another_new) # something need to be fixed
    # print "step control 1", ts_treat.step_control
    # optimizer._change_trust_radius_step(0, 0.25) # change the trust radius
    # print "step control 2", ts_treat.step_control
    # optimizer.find_stepsize_for_latest_point(method="TRIM") 
    # print 'gradient', ts_treat.v_gradient
    # print 'step', ts_treat.stepsize
    # print "dot time value", np.dot(ts_treat.v_gradient, ts_treat.stepsize)
'''
    second_new = optimizer.update_to_new_point_for_latest_point()
    print "norm",np.linalg.norm(second_new.ts_state.gradient_matrix)
    optimizer.add_a_point(second_new)
    second = optimizer._secant_condition(optimizer.points[1], optimizer.points[0])
    print "this is secand",second
    optimizer.update_hessian_for_latest_point(method="SR1")
    print "new hessian", optimizer.points[1].v_hessian
    print "old hessian", optimizer.points[0].v_hessian
    optimizer.tweak_hessian_for_latest_point()
    optimizer.find_stepsize_for_latest_point(method="TRIM")
    third_trial = optimizer.update_to_new_point_for_latest_point()
    print "second stepsize", second_new.step_control
    print "satisfied check",optimizer._check_new_point_satisfied(second_new, third_trial) # something need to be fixed
    optimizer._change_trust_radius_step(1, 0.25) # change the trust radius
    print "new second stepsize", second_new.step_control
    print "stepsize", second_new.stepsize
    test_third = optimizer.update_to_new_point_for_latest_point()
    print "test third"

    third_p = second_new.obtain_new_cc_with_new_delta_v(second_new.stepsize)
    assert np.allclose(third_p.ts_state.coordinates, test_third.ts_state.coordinates)

    print "energy", third_p.ts_state.energy
    print "gradient", third_p.v_gradient
    print "two close tests", np.allclose(third_trial.ts_state.coordinates, third_p.ts_state.coordinates)
    print "third gradient",third_p.ts_state.gradient_matrix
    print "v norm",np.linalg.norm(third_p.v_gradient)
    print optimizer._test_converge(third_p, second_new)
    optimizer.add_a_point(third_p)
    print "new third_p"
    optimizer.update_hessian_for_latest_point(method="SR1")
    print "third hessian ***********"
    print third_p.v_hessian
    optimizer.tweak_hessian_for_latest_point()
    print "third hessian ***********", np.linalg.eigh(third_p.v_hessian)
    print np.linalg.eigh(second_new.v_hessian)
    optimizer.find_stepsize_for_latest_point(method="TRIM")
    print np.linalg.eigh(third_p.v_hessian)[0][1]
    print np.allclose(np.linalg.eigh(third_p.v_hessian)[0][1], 0.005)
    point_four = optimizer.update_to_new_point_for_latest_point()
    print "energy", point_four.ts_state.energy
    print "gradient", point_four.v_gradient
    print "norm",np.linalg.norm(point_four.ts_state.gradient_matrix)
    print "v norm",np.linalg.norm(point_four.v_gradient)
    optimizer.add_a_point(point_four)
    optimizer.update_hessian_for_latest_point(method="SR1")
    optimizer.tweak_hessian_for_latest_point()
    optimizer.find_stepsize_for_latest_point(method="TRIM")
    point_fifth = optimizer.update_to_new_point_for_latest_point()
    print "energy", point_fifth.ts_state.energy
    print "gradient", point_fifth.v_gradient
    print "norm",np.linalg.norm(point_fifth.ts_state.gradient_matrix)
    print "v norm",np.linalg.norm(point_fifth.v_gradient)
    print "hessian", third_p.v_hessian
    print np.linalg.eigh(third_p.v_hessian)
    # print optimizer.points[1].advanced_info['']
    # optimizer.tweak_hessian_for_latest_point()
    # optimizer.find_stepsize_for_latest_point(method="TRIM")
    # third_p = optimizer.update_to_new_point_for_latest_point()
    optimizer.add_a_point(point_fifth)
    optimizer.update_hessian_for_latest_point(method="SR1")
    optimizer.tweak_hessian_for_latest_point()
    optimizer.find_stepsize_for_latest_point(method="TRIM")
    six_th = optimizer.update_to_new_point_for_latest_point()
    print "energy", six_th.ts_state.energy
    print "gradient", six_th.v_gradient
    print "norm",np.linalg.norm(six_th.ts_state.gradient_matrix)
    print "v norm",np.linalg.norm(six_th.v_gradient)
    optimizer.add_a_point(six_th)
    optimizer.update_hessian_for_latest_point(method="SR1")
    optimizer.tweak_hessian_for_latest_point()
    optimizer.find_stepsize_for_latest_point(method="TRIM")
    seven_th = optimizer.update_to_new_point_for_latest_point()
    print "energy", seven_th.ts_state.energy
    print "gradient", seven_th.v_gradient
    print "norm",np.linalg.norm(seven_th.ts_state.gradient_matrix)
    print "v norm",np.linalg.norm(seven_th.v_gradient)
    print "hessian", np.linalg.eigh(six_th.v_hessian)
    optimizer.add_a_point(seven_th)
    optimizer.update_hessian_for_latest_point(method="SR1")
    optimizer.tweak_hessian_for_latest_point()
    optimizer.find_stepsize_for_latest_point(method="TRIM")
    eight_th = optimizer.update_to_new_point_for_latest_point()
    print "energy", eight_th.ts_state.energy
    print "gradient", eight_th.v_gradient
    print "norm",np.linalg.norm(eight_th.ts_state.gradient_matrix)
    print "v norm",np.linalg.norm(eight_th.v_gradient)
    print "hessian", np.linalg.eigh(seven_th.v_hessian)
    optimizer.add_a_point(eight_th)
    optimizer.update_hessian_for_latest_point(method="SR1")
    optimizer.tweak_hessian_for_latest_point()
    optimizer.find_stepsize_for_latest_point(method="TRIM")
    nine_th = optimizer.update_to_new_point_for_latest_point()
    print "energy", nine_th.ts_state.energy
    print "gradient", nine_th.v_gradient
    print "norm",np.linalg.norm(nine_th.ts_state.gradient_matrix)
    print "v norm",np.linalg.norm(nine_th.v_gradient)
    print "hessian", np.linalg.eigh(eight_th.v_hessian)
    optimizer.add_a_point(nine_th)
    optimizer.update_hessian_for_latest_point(method="SR1")
    optimizer.tweak_hessian_for_latest_point()
    optimizer.find_stepsize_for_latest_point(method="TRIM")
    ten_th = optimizer.update_to_new_point_for_latest_point()
    print "energy", ten_th.ts_state.energy
    print "gradient", ten_th.v_gradient
    print "norm",np.linalg.norm(ten_th.ts_state.gradient_matrix)
    print "v norm",np.linalg.norm(ten_th.v_gradient)
    print "hessian", np.linalg.eigh(nine_th.v_hessian)
    optimizer.add_a_point(ten_th)
    optimizer.update_hessian_for_latest_point(method="SR1")
    optimizer.tweak_hessian_for_latest_point()
    optimizer.find_stepsize_for_latest_point(method="TRIM")
    _11_th = optimizer.update_to_new_point_for_latest_point()
    print "energy", _11_th.ts_state.energy
    print "gradient", _11_th.v_gradient
    print "norm",np.linalg.norm(_11_th.ts_state.gradient_matrix)
    print "v norm",np.linalg.norm(_11_th.v_gradient)
    print "hessian", np.linalg.eigh(ten_th.v_hessian)
    optimizer.add_a_point(_11_th)
    '''
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