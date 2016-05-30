import horton as ht
import numpy as np
from saddle.TransitionSearch import *
from saddle.tstreat import TS_Treat
from saddle.optimizer.saddle_optimizer import TrialOptimizer
import pprint
import os


def test_transitionsearch_cl_h_br():
    path = os.path.dirname(os.path.realpath(__file__))
    reactant = ht.IOData.from_file(path + "/Br_HCl.xyz")
    product = ht.IOData.from_file(path + "/Cl_HBr.xyz")
    ts_sample = TransitionSearch(reactant, product)
    ts_sample.auto_ts_search(opt=True, similar=ts_sample.product)
    # print "ic", ts_sample.ts_state.ic
    # print ts_sample.ts_state.coordinates
    # print ts_sample.ts_state.b_matrix
    ts_sample.auto_key_ic_select()
    # length = np.array([0.1, 0, 0])
    # ts_sample.ts_state.use_delta_ic_to_calculate_new_cc(length)
    # print ts_sample.ts_state.coordinates
    # print ts_sample.ts_state.ic
    x_gradient = np.array([0.,  0.,  0.08887695,  0.,  0.,
                           -0.0902661,  0.,  0.,  0.00138915])
    x_hessian = np.array([[0.03011641,  0.,  0., -0.02821553,  0.,
                           0., -0.00190088,  0.,  0.],
                          [0.,  0.03086039,  0.,  0., -0.02863026,
                           0.,  0., -0.00223013,  0.],
                          [0.,  0.,  0.02398155,  0.,  0.,
                           -0.05804879,  0.,  0.,  0.03406724],
                          [-0.02821553,  0.,  0.,  0.02733143,  0.,
                           0.,  0.0008841,  0.,  0.],
                          [0., -0.02863026,  0.,  0.,  0.02756313,
                           0.,  0.,  0.00106714,  0.],
                          [0.,  0., -0.05804879,  0.,  0.,
                           0.10503458,  0.,  0., -0.04698579],
                          [-0.00190088,  0.,  0.,  0.0008841,  0.,
                           0.,  0.00101678,  0.,  0.],
                          [0., -0.00223013,  0.,  0.,  0.00106714,
                           0.,  0.,  0.00116299,  0.],
                          [0.,  0.,  0.03406724,  0.,  0.,
                           -0.04698579,  0.,  0.,  0.01291855]])
    step = np.dot(np.linalg.pinv(x_hessian), x_gradient)
    ts_sample.ts_state.gradient_matrix = x_gradient.copy()
    ts_sample.ts_state.hessian_matrix = x_hessian.copy()
    x_step = -np.dot(np.linalg.pinv(ts_sample.ts_state.hessian_matrix),
                     ts_sample.ts_state.gradient_matrix)
    delta_ic = np.dot(ts_sample.ts_state.b_matrix, x_step)
    ts_sample.ts_state.gradient_x_to_ic()
    ts_sample.ts_state.hessian_x_to_ic()
    ic_step = -np.dot(np.linalg.pinv(ts_sample.ts_state.ic_hessian),
                      ts_sample.ts_state.ic_gradient)
    # # print ic_step
    assert np.allclose(delta_ic, ic_step)

    ts_t = ts_sample.create_ts_treat()
    ts_t.get_new_v_basis()
    ts_t.get_v_gradient()
    ts_t.get_v_hessian()
    # print ts_t.v_gradient
    # print ts_t.v_hessian

    ts_t.set_ic_gradient()
    ts_t.set_ic_hessian()

    v_step = -np.dot(np.linalg.pinv(ts_t.v_hessian), ts_t.v_gradient)
    print "v step", v_step

    new_v_step = v_step.copy()
    new_v_step[0] = 0
    print "new v step", new_v_step

    delta_ic = np.dot(ts_t.v_matrix, v_step)
    print "d ic, ic",delta_ic, ic_step

    new_delta_ic = np.dot(ts_t.v_matrix, new_v_step)
    print "new delta_ic", new_delta_ic

    new_cc_step = ts_t.ts_state._calculate_delta_cc_from_detla_ic(new_delta_ic)
    g = np.dot(ts_t.ts_state.hessian_matrix, new_cc_step.reshape(-1,1))
    print "new cc \n", new_cc_step, g

    cc_step = ts_t.ts_state._calculate_delta_cc_from_detla_ic(ic_step)
    print x_step, cc_step, np.allclose(cc_step, x_step.reshape(-1,3))
    # print ts_t.v_gradient
    # print ts_t.ts_state.ic_gradient


def test_transitionsearch_ch3_h_cl():
    # fn = ht.context.get_fn("../saddle/test/")
    path = os.path.dirname(os.path.realpath(__file__))
    reactant = ht.IOData.from_file(path + "/ch3_hf.xyz")
    product = ht.IOData.from_file(path + "/ch3f_h.xyz")

    # create a object to find best fitting transition state
    ts_sample = TransitionSearch(reactant, product)
    ts_sample.auto_ic_select_combine()
    ts_sample.auto_ts_search(opt=True, similar=ts_sample.product)
    print ts_sample.ts_state.ic
    print ts_sample.ts_state.target_ic
    print ts_sample.reactant.ic
    print ts_sample.product.ic
    print ts_sample.ts_state.procedures
    ts_sample.auto_key_ic_select()
    print ts_sample.ts_state.coordinates
    # to do, use real gradient and hessian
    print new_gradient
    hessian = np.random.rand(18, 18)
    new_hessian = np.dot(hessian.T, hessian)
    ts_sample.ts_state.gradient_matrix = new_gradient
    ts_sample.ts_state.hessian_matrix = new_hessian
    step = - np.dot(np.linalg.inv(new_hessian), new_gradient)
    result = np.dot(new_hessian, step)
    assert np.allclose(result, -new_gradient)

    # print np.linalg.pinv(ts_sample.ts_state.b_matrix)
    # print ts_sample.ts_state.gradient_matrix
    # ts_sample.create_ts_treat()

    # ts_sample.put_transition_state_molucule_in_xyz("xyz_test")
    # print ts_sample.reactant.ic
    # print ts_sample.product.ic
    # print ts_sample.ts_state.target_ic
    # print ts_sample.ts_state.ic
    # print ts_sample.ts_state.procedures
    # print ts_sample.ts_state.numbers
    # print ts_sample.ts_state.b_matrix
    '''
    ts_sample = TransitionSearch(reactant, product)

    # auto select ic for reactant and product in certain way
    ts_sample.auto_ic_select_combine()
    assert np.allclose(ts_sample.reactant.ic, np.array(
        [2.67533253, 5.56209896, 3.14159265]))
    assert np.allclose(ts_sample.product.ic, np.array(
        [5.14254763, 2.45181572, 3.14159265]))
    # auto select proper ic for ts and optimize the initial guess to as close
    # as possible
    ts_sample.auto_ts_search()
    ts_sample.auto_key_ic_select()  # auto select key ic for transition states
    assert abs(ts_sample._ic_key_counter - 2) < 1e-8
    ts_treat = ts_sample.create_ts_treat()
    assert isinstance(ts_treat, TS_Treat)
    print ts_treat.ts_state.coordinates
    cartesian coordinates:
    [[ 0.          0.         -1.29612536]
     [-0.          0.         -5.3030827 ]
     [ 0.          0.          2.61281472]]

    ic coordinates:
    [ 3.90894008, 4.00695734, 3.14159265]

    set the gradient_matrix to be 0.5 on every direction
    
    new_gradient = np.empty(9)
    new_gradient.fill(0.5)
    new_hessian = np.array([[4.44717493,  2.05356967,  1.94014292,  2.53929522,  2.22145167,
                             2.68220307,  3.30788231,  2.40425768,  2.52913864],
                            [2.05356967,  2.7316218,  2.13413339,  1.56308187,  0.99668357,
                             2.07657892,  1.82644009,  1.69260402,  2.20694036],
                            [1.94014292,  2.13413339,  2.96103095,  1.23645561,  0.53444993,
                             2.3399681,  1.8834255,  1.55032916,  2.32321193],
                            [2.53929522,  1.56308187,  1.23645561,  2.13967586,  1.15635668,
                             1.65262475,  1.65608371,  1.13427401,  1.64932224],
                            [2.22145167,  0.99668357,  0.53444993,  1.15635668,  2.14833272,
                             1.43469213,  2.00769528,  1.12468494,  1.28877614],
                            [2.68220307,  2.07657892,  2.3399681,  1.65262475,  1.43469213,
                             3.16509271,  2.40384962,  1.73512774,  2.50605295],
                            [3.30788231,  1.82644009,  1.8834255,  1.65608371,  2.00769528,
                             2.40384962,  3.34313048,  1.76921502,  2.45695502],
                            [2.40425768,  1.69260402,  1.55032916,  1.13427401,  1.12468494,
                             1.73512774,  1.76921502,  1.7635556,  1.71728269],
                            [2.52913864,  2.20694036,  2.32321193,  1.64932224,  1.28877614,
                             2.50605295,  2.45695502,  1.71728269,  2.57545442]])
    ts_treat.ts_state.gradient_matrix = new_gradient
    ts_treat.ts_state.hessian_matrix = new_hessian
    step = - np.dot(np.linalg.inv(new_hessian), new_gradient)
    print step
    step_expectation = [-1.56956788, -0.51902661, -0.15267468,
                        1.19677507, -0.50892484, 0.31015746, 1.2724301, 1.58813772, -1.15669774]
    assert np.allclose(step, step_expectation)
    result = np.dot(new_hessian, step)
    assert np.allclose(result, -new_gradient)

    ts_treat.ts_state.gradient_x_to_ic()
    ts_treat.ts_state.hessian_x_to_ic()
    ts_treat.ts_state.gradient_ic_to_x()
    print ts_treat.ts_state.gradient_matrix

    assert np.allclose(ts_sample.ts_state.ic, [
                       3.90894008, 4.00695734, 3.14159265])
    a_matrix = ts_treat._matrix_a_eigen()
    assert np.allclose(a_matrix, np.linalg.svd(
        ts_treat.ts_state.b_matrix)[0][:, :4])
    b_vector = ts_treat._projection()
    ts_treat.get_v_basis()  # obtain V basis for ts_treat
    ortho_b = TS_Treat.gram_ortho(b_vector)
    dric = np.dot(b_vector, ortho_b)
    new_dric = [dric[:, i] / np.linalg.norm(dric[:, i])
                for i in range(len(dric[0]))]
    new_dric = np.array(new_dric).T
    part1 = np.dot(new_dric, new_dric.T)
    part2 = np.dot(part1, a_matrix)
    nonredu = a_matrix - part2
    ortho_f = TS_Treat.gram_ortho(nonredu)
    rdric = np.dot(nonredu, ortho_f)
    new_rdric = [rdric[:, i] /
                 np.linalg.norm(rdric[:, i]) for i in range(len(rdric[0]))]
    new_rdric = np.array(new_rdric).T
    test_v = np.hstack((new_dric, new_rdric))
    assert np.allclose(ts_treat.v_matrix, test_v)
    '''


if __name__ == '__main__':
    test_transitionsearch_cl_h_br()
