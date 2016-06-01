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
    # assert isinstance(ts_treat, TS_Treat)
    # assert np.allclose(ts_sample.ts_state.target_ic, [
    #                    3.90894008, 4.00695734, 3.14159265])
    # assert np.allclose(ts_sample.ts_state.ic, [
    #                    3.90773539, 4.00847603, 3.14159265])
    # a_matrix = ts_treat._matrix_a_eigen()
    # assert np.allclose(a_matrix, np.linalg.svd(
    #     ts_treat.ts_state.b_matrix)[0][:, :4])
    # b_vector = ts_treat._projection()
    # ts_treat.get_v_basis()  # obtain V basis for ts_treat
    # ortho_b = TS_Treat.gram_ortho(b_vector)
    # dric = np.dot(b_vector, ortho_b)
    # new_dric = [dric[:, i] / np.linalg.norm(dric[:, i])
    #             for i in range(len(dric[0]))]
    # new_dric = np.array(new_dric).T
    # part1 = np.dot(new_dric, new_dric.T)
    # part2 = np.dot(part1, a_matrix)
    # nonredu = a_matrix - part2
    # ortho_f = TS_Treat.gram_ortho(nonredu)
    # rdric = np.dot(nonredu, ortho_f)
    # new_rdric = [rdric[:, i] /
    #              np.linalg.norm(rdric[:, i]) for i in range(len(rdric[0]))]
    # new_rdric = np.array(new_rdric).T
    # test_v = np.hstack((new_dric, new_rdric))
    # assert np.allclose(ts_treat.v_matrix, test_v)

    "try to specify the ic manualy"

    new_ts_sample = TransitionSearch(reactant, product)
    new_ts_sample.add_bond(0, 1, "regular", [new_ts_sample.reactant, new_ts_sample.product])
    new_ts_sample.add_bond(0, 2, "regular", [new_ts_sample.reactant, new_ts_sample.product])
    new_ts_sample.add_bond(1, 2, "regular", [new_ts_sample.reactant, new_ts_sample.product])
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
    optimizer.add_a_point(ts_treat)
    # print "hessian, initial\n", ts_treat.v_hessian,"\n",
    # np.linalg.eigh(ts_treat.v_hessian)
    optimizer.initialize_trm_for_point_with_index(0)
    # optimizer.tweak_hessian_for_latest_point()
    optimizer.find_stepsize_for_latest_point(method="TRIM")
    p_2 = optimizer.update_to_new_point_for_latest_point(True, method='gs')
    # print "-------"
    # print p_2.ts_state.energy
    # print "p2", p_2.v_gradient
    # veri = optimizer.verify_new_point_with_latest_point(p_2)
    # print "test gradient",veri
    optimizer.add_a_point(p_2)
    optimizer.update_trust_radius_latest_point(method='gradient')
    # print "finite test", optimizer._test_necessity_for_finite_difference(1)
    optimizer.tweak_hessian_for_latest_point()
    optimizer.find_stepsize_for_latest_point(method="TRIM")
    p_3 = optimizer.update_to_new_point_for_latest_point(True, method='gs')
    # print "-------"
    # print "p3",p_3.ts_state.energy, p_3.v_gradient
    veri = optimizer.verify_new_point_with_latest_point(p_3)
    # if not veri:
    #     optimizer.find_stepsize_for_latest_point(method='TRIM')
    #     p_3 = optimizer.update_to_new_point_for_latest_point(True, method='gs')
    # print "-------"
    # print "p3 new",p_3_new.ts_state.energy, p_3_new.v_gradient,
    # p_2.step_control
    optimizer.add_a_point(p_3)
    # optimizer.update_trust_radius_latest_point(method='gradient')
    # #optimizer.tweak_hessian_for_latest_point()
    # optimizer.find_stepsize_for_latest_point(method="TRIM")
    # p_4 = optimizer.update_to_new_point_for_latest_point(True, method='gs')
    # print "-------"
    # print "p4",p_4.ts_state.energy, p_4.v_gradient
    # veri = optimizer.verify_new_point_with_latest_point(p_4)
    # if not veri:
    #     optimizer.find_stepsize_for_latest_point(method='TRIM')
    #     p_4 = optimizer.update_to_new_point_for_latest_point(True, method='gs')
    # print "-------"
    # print "p4 new",p_4.ts_state.energy, p_4.v_gradient, p_3_new.step_control
    # optimizer.add_a_point(p_4)
    # optimizer.update_trust_radius_latest_point(method='gradient')
    # optimizer.tweak_hessian_for_latest_point()
    # optimizer.find_stepsize_for_latest_point(method="TRIM")
    # p_5 = optimizer.update_to_new_point_for_latest_point(True, method='gs')
    # print "-------"
    # print "p5",p_5.ts_state.energy, p_5.v_gradient, p_4.step_control
    # veri = optimizer.verify_new_point_with_latest_point(p_5)
    # if not veri:
    #     optimizer.find_stepsize_for_latest_point(method='TRIM')
    #     p_5 = optimizer.update_to_new_point_for_latest_point(True, method='gs')
    # print "-------"
    # print "p5 new",p_5.ts_state.energy, p_5.v_gradient, p_4.step_control
    # optimizer.add_a_point(p_5)
    # optimizer.update_trust_radius_latest_point(method='gradient')
    # optimizer.tweak_hessian_for_latest_point()
    # optimizer.find_stepsize_for_latest_point(method="TRIM")
    # p_6 = optimizer.update_to_new_point_for_latest_point(True, method='gs')
    # print "-------"
    # print "p6",p_6.ts_state.energy, p_6.v_gradient, p_5.step_control
    # veri = optimizer.verify_new_point_with_latest_point(p_6)
    # if not veri:
    #     optimizer.find_stepsize_for_latest_point(method='TRIM')
    #     p_6 = optimizer.update_to_new_point_for_latest_point(True, method='gs')
    #     print "-------"
    #     print "p6 new",p_6.ts_state.energy, p_6.v_gradient, p_5.step_control
    # optimizer.add_a_point(p_6)
    # optimizer.update_trust_radius_latest_point(method='gradient')
    # optimizer.tweak_hessian_for_latest_point()
    # optimizer.find_stepsize_for_latest_point(method="TRIM")
    # p_7 = optimizer.update_to_new_point_for_latest_point(True, method='gs')
    # print "-------"
    # print "p7",p_7.ts_state.energy, p_7.v_gradient, p_6.step_control
    # veri = optimizer.verify_new_point_with_latest_point(p_7)
    # if not veri:
    #     optimizer.find_stepsize_for_latest_point(method='TRIM')
    #     p_7 = optimizer.update_to_new_point_for_latest_point(True, method='gs')
    #     print "-------"
    #     print "p7 new",p_7.ts_state.energy, p_7.v_gradient, p_6.step_control
    # optimizer.add_a_point(p_7)
    # optimizer.update_trust_radius_latest_point(method='gradient')
    # optimizer.tweak_hessian_for_latest_point()
    # optimizer.find_stepsize_for_latest_point(method="TRIM")
    # p_8 = optimizer.update_to_new_point_for_latest_point(True, method='gs')
    # print "-------"
    # print "p8",p_8.ts_state.energy, p_8.v_gradient, p_7.step_control
    # veri = optimizer.verify_new_point_with_latest_point(p_8)
    # if not veri:
    #     optimizer.find_stepsize_for_latest_point(method='TRIM')
    #     p_8 = optimizer.update_to_new_point_for_latest_point(True, method='gs')
    #     print "-------"
    #     print "p8",p_8.ts_state.energy, p_8.v_gradient, p_7.step_control
    # optimizer.add_a_point(p_8)
    # optimizer.update_trust_radius_latest_point(method='gradient')
    # optimizer.tweak_hessian_for_latest_point()
    # optimizer.find_stepsize_for_latest_point(method="TRIM")
    # p_9 = optimizer.update_to_new_point_for_latest_point(True, method='gs')
    # print "-------"
    # print "p9",p_9.ts_state.energy, p_9.v_gradient, p_8.step_control
    # veri = optimizer.verify_new_point_with_latest_point(p_9)
    # if not veri:
    #     optimizer.find_stepsize_for_latest_point(method='TRIM')
    #     p_9 = optimizer.update_to_new_point_for_latest_point(True, method='gs')
    #     print "-------"
    #     print "p9 new",p_9.ts_state.energy, p_9.v_gradient, p_8.step_control
    # optimizer.add_a_point(p_9)
    # optimizer.update_trust_radius_latest_point(method='gradient')
    # optimizer.tweak_hessian_for_latest_point()
    # optimizer.find_stepsize_for_latest_point(method="TRIM")
    # p_10 = optimizer.update_to_new_point_for_latest_point(True, method='gs')
    # print "-------"
    # print "p10",p_10.ts_state.energy, p_10.v_gradient, p_9.step_control
    # veri = optimizer.verify_new_point_with_latest_point(p_10)
    # if not veri:
    #     optimizer.find_stepsize_for_latest_point(method='TRIM')
    #     p_10 = optimizer.update_to_new_point_for_latest_point(True, method='gs')
    #     print "-------"
    #     print "p10 new",p_10.ts_state.energy, p_10.v_gradient, p_9.step_control
    # optimizer.add_a_point(p_10)
    # optimizer.update_trust_radius_latest_point(method='gradient')
    # optimizer.tweak_hessian_for_latest_point()
    # optimizer.find_stepsize_for_latest_point(method="TRIM")
    # p_11 = optimizer.update_to_new_point_for_latest_point(True, method='gs')
    # print "-------"
    # print "p11",p_11.ts_state.energy, p_11.v_gradient, p_10.step_control
    # veri = optimizer.verify_new_point_with_latest_point(p_11)
    # if not veri:
    #     optimizer.find_stepsize_for_latest_point(method='TRIM')
    #     p_11 = optimizer.update_to_new_point_for_latest_point(True, method='gs')
    #     print "-------"
    #     print "p11 new",p_11.ts_state.energy, p_11.v_gradient, p_10.step_control
    # optimizer.add_a_point(p_11)
    # optimizer.update_trust_radius_latest_point(method='gradient')
    # optimizer.tweak_hessian_for_latest_point()
    # optimizer.find_stepsize_for_latest_point(method="TRIM")
    # p_12 = optimizer.update_to_new_point_for_latest_point(True, method='gs')
    # print "-------"
    # print "p12",p_12.ts_state.energy, p_12.v_gradient, p_11.step_control
    # veri = optimizer.verify_new_point_with_latest_point(p_12)
    # if not veri:
    #     optimizer.find_stepsize_for_latest_point(method='TRIM')
    #     p_12 = optimizer.update_to_new_point_for_latest_point(True, method='gs')
    #     print "-------"
    #     print "p12 new",p_12.ts_state.energy, p_12.v_gradient, p_11.step_control
    # optimizer.add_a_point(p_12)
    # optimizer.update_trust_radius_latest_point(method='gradient')
    # optimizer.tweak_hessian_for_latest_point()
    # optimizer.find_stepsize_for_latest_point(method="TRIM")
    # p_13 = optimizer.update_to_new_point_for_latest_point(True, method='gs')
    # print "-------"
    # print "p13",p_13.ts_state.energy, p_13.v_gradient, p_12.step_control
    # veri = optimizer.verify_new_point_with_latest_point(p_13)
    # if not veri:
    #     optimizer.find_stepsize_for_latest_point(method='TRIM')
    #     p_13 = optimizer.update_to_new_point_for_latest_point(True, method='gs')
    #     print "-------"
    # print "p13 new",p_13.ts_state.energy, p_13.v_gradient, p_12.step_control

    '''
    optimizer.update_hessian_for_latest_point(method='SR1')
    print "p_2 hessian \n", p_2.v_hessian, np.linalg.eigh(p_2.v_hessian)
    
    hveri = optimizer._test_necessity_for_finite_difference(1)
    print "test finite diff", hveri
    optimizer._update_hessian_finite_difference(1, hveri, "gs", 0.001)
    print "p_2 hessian \n", p_2.v_hessian
    optimizer.update_trust_radius_latest_point(method='gradient')
    optimizer.tweak_hessian_for_latest_point()
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    p_3 = optimizer.update_to_new_point_for_latest_point(method='gs')
    print "p3", p_3.v_gradient, p_3.ts_state.energy 
    print "step,",p_2.step_control, p_2.stepsize, np.linalg.norm(p_2.stepsize)
    veri = optimizer.verify_new_point_with_latest_point(p_3)
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    print "step,",veri, p_2.step_control, p_2.stepsize, np.linalg.norm(p_2.stepsize)
    p_3_new = optimizer.update_to_new_point_for_latest_point(method='gs')
    print "-------"
    print "p3_new", p_3_new.v_gradient, p_3_new.ts_state.energy
    optimizer.add_a_point(p_3_new)
    optimizer.update_hessian_for_latest_point(method='SR1')
    hveri = optimizer._test_necessity_for_finite_difference(2)
    print hveri
    optimizer._update_hessian_finite_difference(2, hveri, "gs", 0.001)
    optimizer.tweak_hessian_for_latest_point()
    optimizer.find_stepsize_for_latest_point(method="TRIM")
    p_4 = optimizer.update_to_new_point_for_latest_point(method="gs")
    print p_4.step_control
    print 'p4', p_4.v_gradient, p_4.ts_state.energy
    print 'step', p_3_new.stepsize, p_3_new.step_control, np.linalg.norm(p_3_new.stepsize)
    veri = optimizer.verify_new_point_with_latest_point(p_4)
    print veri
    optimizer.find_stepsize_for_latest_point(method="TRIM")
    print "step", p_3_new.stepsize, p_3_new.step_control, np.linalg.norm(p_3_new.stepsize)
    p_4_new = optimizer.update_to_new_point_for_latest_point(method='gs')
    print "-------"
    print 'p4new', p_4_new.v_gradient, p_4_new.ts_state.energy
    optimizer.add_a_point(p_4_new)
    optimizer.update_hessian_for_latest_point(method='SR1')
    hveri = optimizer._test_necessity_for_finite_difference(3)
    hveri = [0, 1, 2]
    optimizer._update_hessian_finite_difference(3, hveri, "gs", 0.001)
    print "eigen values", np.linalg.eigh(p_4_new.v_hessian)
    optimizer.tweak_hessian_for_latest_point()
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    p_5 = optimizer.update_to_new_point_for_latest_point(method='gs')
    print "step p4 new old",p_4_new.stepsize, p_4_new.step_control
    print p_5.v_gradient, p_5.ts_state.energy, p_5.ts_state.gradient_matrix
    optimizer.verify_new_point_with_latest_point(p_5)
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    p_5_new = optimizer.update_to_new_point_for_latest_point(method='gs')
    print "-------"
    print "p5", p_5_new.v_gradient, p_5_new.ts_state.energy, p_5_new.ts_state.gradient_matrix
    print np.linalg.eigh(p_4_new.v_hessian)
    optimizer.add_a_point(p_5_new)
    optimizer.update_hessian_for_latest_point(method="SR1")
    hveri = optimizer._test_necessity_for_finite_difference(4)
    optimizer._update_hessian_finite_difference(4, hveri, "gs", 0.001)
    optimizer.tweak_hessian_for_latest_point()
    optimizer.find_stepsize_for_latest_point(method="TRIM")
    p_6 = optimizer.update_to_new_point_for_latest_point(method='gs')
    print 'p6', p_6.v_gradient, p_6.ts_state.energy
    result = optimizer.verify_new_point_with_latest_point(p_6)
    if not result:
        optimizer.find_stepsize_for_latest_point(method="TRIM")
        p_6_new = optimizer.update_to_new_point_for_latest_point(method='gs')
        print 'p6_new', p_6_new.v_gradient, p_6_new.ts_state.energy 
    optimizer.add_a_point(p_6_new)
    optimizer.update_hessian_for_latest_point(method="SR1")
    optimizer.tweak_hessian_for_latest_point()
    hveri = optimizer._test_necessity_for_finite_difference(5)
    optimizer._update_hessian_finite_difference(5, hveri, "gs", 0.001)
    optimizer.find_stepsize_for_latest_point(method="TRIM")
    p_7 = optimizer.update_to_new_point_for_latest_point(method='gs')
    print 'p7', p_7.v_gradient, p_7.ts_state.energy
    result = optimizer.verify_new_point_with_latest_point(p_7)
    if not result:
        optimizer.find_stepsize_for_latest_point(method="TRIM")
        p_7_new = optimizer.update_to_new_point_for_latest_point(method='gs')
        print 'p7_new', p_7_new.v_gradient, p_7_new.ts_state.energy
    optimizer.add_a_point(p_7_new)
    optimizer.update_hessian_for_latest_point(method="SR1")
    optimizer.tweak_hessian_for_latest_point()
    hveri = optimizer._test_necessity_for_finite_difference(6)
    optimizer._update_hessian_finite_difference(6, hveri, "gs", 0.001)
    optimizer.find_stepsize_for_latest_point(method="TRIM")
    p_8 = optimizer.update_to_new_point_for_latest_point(method='gs')
    print 'p8', p_8.v_gradient, p_8.ts_state.energy 
    '''
    '''
    ts_treat.ts_state.get_energy_gradient_hessian() #obtain energy, gradient, and hessian
    ts_treat.ts_state.get_energy_gradient()
    # print "hessian", ts_treat.ts_state.hessian_matrix
    print "first obtain energy and gradient", ts_treat.ts_state.energy, ts_treat.ts_state.gradient_matrix#, ts_treat.ts_state.hessian_matrix
    ts_treat.get_v_gradient()
    ts_treat.get_v_hessian()
    # print "symmetry" ,ts_treat.v_hessian
    print ts_treat.ts_state.procedures
    print ts_treat.v_hessian.shape
    print "hessian", np.linalg.eigh(ts_treat.v_hessian)
    print ts_treat.v_gradient
    print ts_treat.ts_state.energy
    print ts_treat.ts_state.gradient_matrix
    print ts_treat.ts_state.coordinates
    print "start optimization-------------------"
    optimizer = TrialOptimizer()
    optimizer.set_trust_radius_method(method="default", parameter=3)
    optimizer.add_a_point(ts_treat)
    print ts_treat.v_hessian,"\n, coor for the 1st", ts_treat.ts_state.coordinates
    optimizer.initialize_trm_for_point_with_index(0)
    optimizer.tweak_hessian_for_latest_point()
    optimizer.find_stepsize_for_latest_point(method="TRIM")
    _2_p = optimizer.update_to_new_point_for_latest_point()
    print _2_p.v_gradient
    veri = optimizer.verify_new_point_with_latest_point(_2_p)
    print veri
    optimizer.find_stepsize_for_latest_point(method="TRIM")
    _2_p_new = optimizer.update_to_new_point_for_latest_point()
    print "2 gradient",_2_p_new.v_gradient, np.linalg.norm(_2_p_new.v_gradient), np.linalg.norm(_2_p_new.ts_state.gradient_matrix)
    optimizer.add_a_point(_2_p_new)
    optimizer.update_hessian_for_latest_point(method='SR1')
    print _2_p_new.v_hessian
    hveri = optimizer._test_necessity_for_finite_difference(1)
    print hveri
    optimizer._update_hessian_finite_difference(1, hveri)
    hveri = optimizer._test_necessity_for_finite_difference(1)
    print "hessian",_2_p_new.v_hessian
    print hveri
    optimizer.update_trust_radius_latest_point(method='gradient')
    optimizer.tweak_hessian_for_latest_point()
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    _3_p = optimizer.update_to_new_point_for_latest_point()
    print "3 gradient",_3_p.v_gradient, np.linalg.norm(_3_p.v_gradient), np.linalg.norm(_3_p.ts_state.gradient_matrix)
    veri = optimizer.verify_new_point_with_latest_point(_3_p)
    optimizer.find_stepsize_for_latest_point(method="TRIM")
    _3_p_new = optimizer.update_to_new_point_for_latest_point()
    print "3 gradient",_3_p_new.v_gradient, np.linalg.norm(_3_p_new.v_gradient), np.linalg.norm(_3_p_new.ts_state.gradient_matrix)
    optimizer.add_a_point(_3_p_new)
    optimizer.update_hessian_for_latest_point(method='SR1')
    veri = optimizer._test_necessity_for_finite_difference(2)
    print veri
    optimizer._update_hessian_finite_difference(2, veri)
    print "3 hessian",np.linalg.eigh(_3_p_new.v_hessian)
    optimizer.update_trust_radius_latest_point(method='gradient')
    optimizer.tweak_hessian_for_latest_point()
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    _4_p = optimizer.update_to_new_point_for_latest_point()
    print "4 gradient", _4_p.v_gradient,np.linalg.norm(_4_p.ts_state.gradient_matrix)
    veri = optimizer.verify_new_point_with_latest_point(_4_p)
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    _4_p_new = optimizer.update_to_new_point_for_latest_point()
    print "4 new gradient", _4_p_new.v_gradient,np.linalg.norm(_4_p_new.ts_state.gradient_matrix),"\n", _4_p_new.ts_state.coordinates
    optimizer.add_a_point(_4_p_new)
    optimizer.update_hessian_for_latest_point(method='SR1')
    veri = optimizer._test_necessity_for_finite_difference(3)
    optimizer._update_hessian_finite_difference(3, veri)
    print "4 hessian",np.linalg.eigh(_4_p_new.v_hessian)
    optimizer.update_trust_radius_latest_point(method='gradient')
    optimizer.tweak_hessian_for_latest_point()
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    _5_p = optimizer.update_to_new_point_for_latest_point()
    print "5 gradient", _5_p.v_gradient,np.linalg.norm(_5_p.ts_state.gradient_matrix)
    veri = optimizer.verify_new_point_with_latest_point(_5_p)
    print veri, _5_p.ts_state.coordinates
    optimizer.add_a_point(_5_p)
    result = optimizer.verify_convergence_for_latest_point()
    print result
    optimizer.update_hessian_for_latest_point(method='SR1')
    print _4_p_new.v_hessian, "\n",np.linalg.eigh(_4_p_new.v_hessian)
    print _5_p.v_hessian, "\n",np.linalg.eigh(_5_p.v_hessian)
    veri = optimizer._test_necessity_for_finite_difference(4)
    print veri
    optimizer._update_hessian_finite_difference(4, veri)
    print "5 hessian",np.linalg.eigh(_5_p.v_hessian)
    optimizer.update_trust_radius_latest_point(method='gradient')
    optimizer.tweak_hessian_for_latest_point()
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    _6_p = optimizer.update_to_new_point_for_latest_point()
    print "6 gradient", _6_p.v_gradient,np.linalg.norm(_6_p.ts_state.gradient_matrix)
    veri = optimizer.verify_new_point_with_latest_point(_6_p)
    print veri
    optimizer.add_a_point(_6_p)
    result = optimizer.verify_convergence_for_latest_point()
    print result
    optimizer.update_hessian_for_latest_point(method='SR1')
    veri = optimizer._test_necessity_for_finite_difference(5)
    optimizer._update_hessian_finite_difference(5, veri)
    print "6 hessian",np.linalg.eigh(_6_p.v_hessian)
    optimizer.update_trust_radius_latest_point(method='gradient')
    optimizer.tweak_hessian_for_latest_point()
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    print "6 hessian",np.linalg.eigh(_6_p.v_hessian)
    _7_p = optimizer.update_to_new_point_for_latest_point()
    print "7 gradient", _7_p.v_gradient,np.linalg.norm(_7_p.ts_state.gradient_matrix)
    veri = optimizer.verify_new_point_with_latest_point(_7_p)
    print veri
    optimizer.add_a_point(_7_p)
    result = optimizer.verify_convergence_for_latest_point()
    print result
    optimizer.update_hessian_for_latest_point(method='SR1')
    veri = optimizer._test_necessity_for_finite_difference(6)
    print veri
    optimizer._update_hessian_finite_difference(6, veri)
    print "7 hessian old",_7_p.v_hessian, "\n",np.linalg.eigh(_7_p.v_hessian)
    # veri.append(0)
    optimizer._update_hessian_finite_difference(6, veri)
    print "7 hessian new",_7_p.v_hessian, "\n",np.linalg.eigh(_7_p.v_hessian)
    optimizer.update_trust_radius_latest_point(method='gradient')
    optimizer.tweak_hessian_for_latest_point()
    print "7 hessian",np.linalg.eigh(_7_p.v_hessian)
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    _8_p = optimizer.update_to_new_point_for_latest_point()
    print "8 gradient", _8_p.v_gradient,np.linalg.norm(_8_p.ts_state.gradient_matrix)
    veri = optimizer.verify_new_point_with_latest_point(_8_p)
    print veri
    _8_p_new = optimizer.update_to_new_point_for_latest_point()
    optimizer.add_a_point(_8_p_new)
    result = optimizer.verify_convergence_for_latest_point()
    print result
    optimizer.update_hessian_for_latest_point(method='SR1')
    veri = optimizer._test_necessity_for_finite_difference(7)
    print "finite", veri
    # optimizer._update_hessian_finite_difference(7, veri)
    print "8 hessian old",_8_p_new.v_hessian, "\n",np.linalg.eigh(_8_p_new.v_hessian)
    # veri.append(0)
    # optimizer._update_hessian_finite_difference(7, veri)
    # print "8 hessian new",_8_p_new.v_hessian, "\n",np.linalg.eigh(_8_p_new.v_hessian)
    optimizer.update_trust_radius_latest_point(method='gradient')
    optimizer.tweak_hessian_for_latest_point()
    print "8 hessian",np.linalg.eigh(_8_p_new.v_hessian)
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    _9_p = optimizer.update_to_new_point_for_latest_point()
    print "9 gradient", _9_p.v_gradient,np.linalg.norm(_9_p.ts_state.gradient_matrix)

    veri = optimizer.verify_new_point_with_latest_point(_9_p)
    print veri
    optimizer.add_a_point(_9_p)
    result = optimizer.verify_convergence_for_latest_point()
    print result
    optimizer.update_hessian_for_latest_point(method='SR1')
    veri = optimizer._test_necessity_for_finite_difference(8)
    print veri
    # print "8 hessian old",_8_p_new.v_hessian, "\n",np.linalg.eigh(_8_p_new.v_hessian)
    optimizer.update_trust_radius_latest_point(method='gradient')
    optimizer.tweak_hessian_for_latest_point()
    # print "8 hessian",np.linalg.eigh(_8_p_new.v_hessian)
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    _10_p = optimizer.update_to_new_point_for_latest_point()
    print "10 gradient", _10_p.v_gradient,np.linalg.norm(_10_p.ts_state.gradient_matrix)

    veri = optimizer.verify_new_point_with_latest_point(_10_p)
    print veri
    optimizer.add_a_point(_10_p)
    result = optimizer.verify_convergence_for_latest_point()
    print result
    optimizer.update_hessian_for_latest_point(method='SR1')
    veri = optimizer._test_necessity_for_finite_difference(9)
    print veri
    # print "8 hessian old",_8_p_new.v_hessian, "\n",np.linalg.eigh(_8_p_new.v_hessian)
    optimizer.tweak_hessian_for_latest_point()
    optimizer.update_trust_radius_latest_point(method='gradient')
    print "11 hessian",np.linalg.eigh(_10_p.v_hessian)
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    _11_p = optimizer.update_to_new_point_for_latest_point()
    print "11 gradient", _11_p.v_gradient,np.linalg.norm(_11_p.ts_state.gradient_matrix)
    
    veri = optimizer.verify_new_point_with_latest_point(_11_p)
    print veri
    optimizer.add_a_point(_11_p)
    result = optimizer.verify_convergence_for_latest_point()
    print result
    optimizer.update_hessian_for_latest_point(method='SR1')
    veri = optimizer._test_necessity_for_finite_difference(10)
    print veri
    # print "8 hessian old",_8_p_new.v_hessian, "\n",np.linalg.eigh(_8_p_new.v_hessian)
    optimizer.tweak_hessian_for_latest_point()
    optimizer.update_trust_radius_latest_point(method='gradient')
    print "12 hessian",np.linalg.eigh(_11_p.v_hessian)
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    _12_p = optimizer.update_to_new_point_for_latest_point()
    print "12 gradient", _12_p.v_gradient,np.linalg.norm(_12_p.ts_state.gradient_matrix)

    veri = optimizer.verify_new_point_with_latest_point(_12_p)
    print veri
    optimizer.add_a_point(_12_p)
    result = optimizer.verify_convergence_for_latest_point()
    print result
    optimizer.update_hessian_for_latest_point(method='SR1')
    veri = optimizer._test_necessity_for_finite_difference(11)
    print veri
    # print "8 hessian old",_8_p_new.v_hessian, "\n",np.linalg.eigh(_8_p_new.v_hessian)
    optimizer.tweak_hessian_for_latest_point()
    optimizer.update_trust_radius_latest_point(method='gradient')
    print "13 hessian",np.linalg.eigh(_12_p.v_hessian)
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    _13_p = optimizer.update_to_new_point_for_latest_point()
    print "13 gradient", _13_p.v_gradient,np.linalg.norm(_13_p.ts_state.gradient_matrix)

    veri = optimizer.verify_new_point_with_latest_point(_13_p)
    print veri
    optimizer.add_a_point(_13_p)
    result = optimizer.verify_convergence_for_latest_point()
    print result
    optimizer.update_hessian_for_latest_point(method='SR1')
    veri = optimizer._test_necessity_for_finite_difference(12)
    print veri
    # print "8 hessian old",_8_p_new.v_hessian, "\n",np.linalg.eigh(_8_p_new.v_hessian)
    optimizer.tweak_hessian_for_latest_point()
    optimizer.update_trust_radius_latest_point(method='gradient')
    print "13 hessian",np.linalg.eigh(_13_p.v_hessian)
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    _14_p = optimizer.update_to_new_point_for_latest_point()
    print "14 gradient", _14_p.v_gradient,np.linalg.norm(_14_p.ts_state.gradient_matrix), _14_p.ts_state.ic

    veri = optimizer.verify_new_point_with_latest_point(_14_p)
    print "resize", veri
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    _14_p_new = optimizer.update_to_new_point_for_latest_point()
    optimizer.add_a_point(_14_p_new)
    result = optimizer.verify_convergence_for_latest_point()
    print result
    optimizer.update_hessian_for_latest_point(method='SR1')
    veri = optimizer._test_necessity_for_finite_difference(13)
    print veri
    # print "8 hessian old",_14_p_new.v_hessian, "\n",np.linalg.eigh(_14_p_new.v_hessian)
    optimizer.tweak_hessian_for_latest_point()
    optimizer.update_trust_radius_latest_point(method='gradient')
    print "14 hessian",np.linalg.eigh(_14_p_new.v_hessian)
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    _15_p = optimizer.update_to_new_point_for_latest_point()
    print "15 gradient", _15_p.v_gradient,np.linalg.norm(_15_p.ts_state.gradient_matrix),_15_p.ts_state.ic

    veri = optimizer.verify_new_point_with_latest_point(_15_p)
    print "resize", veri
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    _15_p_new = optimizer.update_to_new_point_for_latest_point()
    optimizer.add_a_point(_15_p_new)
    result = optimizer.verify_convergence_for_latest_point()
    print result
    optimizer.update_hessian_for_latest_point(method='SR1')
    veri = optimizer._test_necessity_for_finite_difference(13)
    print veri
    # print "8 hessian old",_15_p_new.v_hessian, "\n",np.linalg.eigh(_14_p_new.v_hessian)
    optimizer.tweak_hessian_for_latest_point()
    optimizer.update_trust_radius_latest_point(method='gradient')
    print "15 hessian",np.linalg.eigh(_15_p_new.v_hessian)
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    _16_p = optimizer.update_to_new_point_for_latest_point()
    print "16 gradient", _16_p.v_gradient,np.linalg.norm(_16_p.ts_state.gradient_matrix), _16_p.ts_state.ic

    veri = optimizer.verify_new_point_with_latest_point(_16_p)
    print "resize", veri
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    _16_p_new = optimizer.update_to_new_point_for_latest_point()
    optimizer.add_a_point(_16_p_new)
    result = optimizer.verify_convergence_for_latest_point()
    print result
    optimizer.update_hessian_for_latest_point(method='SR1')
    veri = optimizer._test_necessity_for_finite_difference(14)
    print veri
    # print "8 hessian old",_16_p_new.v_hessian, "\n",np.linalg.eigh(_14_p_new.v_hessian)
    optimizer.tweak_hessian_for_latest_point()
    optimizer.update_trust_radius_latest_point(method='gradient')
    print "15 hessian",np.linalg.eigh(_16_p_new.v_hessian)
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    _17_p = optimizer.update_to_new_point_for_latest_point()
    print "16 gradient", _17_p.v_gradient,np.linalg.norm(_17_p.ts_state.gradient_matrix), _17_p.ts_state.ic

    veri = optimizer.verify_new_point_with_latest_point(_17_p)
    print "resize", veri
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    _17_p_new = optimizer.update_to_new_point_for_latest_point()
    optimizer.add_a_point(_17_p_new)
    result = optimizer.verify_convergence_for_latest_point()
    print result
    optimizer.update_hessian_for_latest_point(method='SR1')
    veri = optimizer._test_necessity_for_finite_difference(15)
    print veri
    # print "8 hessian old",_17_p_new.v_hessian, "\n",np.linalg.eigh(_14_p_new.v_hessian)
    optimizer.tweak_hessian_for_latest_point()
    optimizer.update_trust_radius_latest_point(method='gradient')
    print "15 hessian",np.linalg.eigh(_17_p_new.v_hessian)
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    _18_p = optimizer.update_to_new_point_for_latest_point()
    print "17 gradient", _18_p.v_gradient,np.linalg.norm(_18_p.ts_state.gradient_matrix), _18_p.ts_state.ic

    veri = optimizer.verify_new_point_with_latest_point(_18_p)
    print "resize", veri
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    _18_p_new = optimizer.update_to_new_point_for_latest_point()
    optimizer.add_a_point(_18_p_new)
    result = optimizer.verify_convergence_for_latest_point()
    print result
    optimizer.update_hessian_for_latest_point(method='SR1')
    veri = optimizer._test_necessity_for_finite_difference(16)
    print veri
    # print "8 hessian old",_18_p_new.v_hessian, "\n",np.linalg.eigh(_14_p_new.v_hessian)
    optimizer.tweak_hessian_for_latest_point()
    optimizer.update_trust_radius_latest_point(method='gradient')
    print "15 hessian",np.linalg.eigh(_18_p_new.v_hessian)
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    _19_p = optimizer.update_to_new_point_for_latest_point()
    print "17 gradient", _19_p.v_gradient,np.linalg.norm(_19_p.ts_state.gradient_matrix), _19_p.ts_state.ic

    # veri = optimizer.verify_new_point_with_latest_point(_19_p)
    print "resize", veri
    # optimizer.find_stepsize_for_latest_point(method='TRIM')
    # _19_p_new = optimizer.update_to_new_point_for_latest_point()
    optimizer.add_a_point(_19_p)
    result = optimizer.verify_convergence_for_latest_point()
    print result
    optimizer.update_hessian_for_latest_point(method='SR1')
    veri = optimizer._test_necessity_for_finite_difference(17)
    print veri
    # print "8 hessian old",_19_p_new.v_hessian, "\n",np.linalg.eigh(_14_p_new.v_hessian)
    optimizer.tweak_hessian_for_latest_point()
    optimizer.update_trust_radius_latest_point(method='gradient')
    print "15 hessian",np.linalg.eigh(_19_p.v_hessian)
    optimizer.find_stepsize_for_latest_point(method='TRIM')
    _20_p = optimizer.update_to_new_point_for_latest_point()
    print "17 gradient", _20_p.v_gradient,np.linalg.norm(_20_p.ts_state.gradient_matrix), _20_p.ts_state.ic
    '''
    # label

    # optimizer.update_hessian_for_latest_point(method='SR1')
    # hveri = optimizer._test_necessity_for_finite_difference(2)
    # print hveri
    # optimizer._update_hessian_finite_difference(2, hveri)
    # hveri = optimizer._test_necessity_for_finite_difference(2)

    # optimizer.

    # step = -np.dot(np.linalg.pinv(ts_treat.v_hessian), ts_treat.v_gradient)
    # print "step",step
    # ts_treat._diagnolize_h_matrix()
    # print "eigen",ts_treat.advanced_info['eigenvalues']
    # result = np.zeros(ts_treat.ts_state.dof)
    # for i in range(ts_treat.ts_state.dof):
    #     print i
    #     p1 = np.dot(ts_treat.advanced_info['eigenvectors'][:,i].T, ts_treat.v_gradient)
    #     if np.allclose(ts_treat.advanced_info['eigenvalues'][i],0):
    #         result += np.zeros(ts_treat.ts_state.dof)
    #     else:
    #         p1 /= ts_treat.advanced_info['eigenvalues'][i]
    #         p2 = np.dot(p1, ts_treat.advanced_info['eigenvectors'][:,i])
    #         result += p2
    # print result

    # ts_treat.stepsize = step
    # print "norm",np.linalg.norm(ts_treat.v_hessian)
    # print "norm",np.linalg.norm(ts_treat.ts_state.hessian_matrix)
    # original = optimizer.update_to_new_point_for_latest_point()
    # print original.ts_state.energy
    # original.ts_state.get_energy_gradient_hessian()
    # original.get_v_gradient()
    # original.get_v_hessian()
    # print original.v_hessian
    # print "norm",np.linalg.norm(original.v_hessian)
    # print "norm",np.linalg.norm(original.ts_state.hessian_matrix)
    # print np.linalg.eigh(original.v_hessian)
    # optimizer.add_a_point(original)
    # optimizer.update_hessian_for_latest_point(method="SR1")
    # print original.v_hessian
    # print optimizer._test_necessity_for_finite_difference(1)
    # optimizer._update_hessian_finite_difference(1)
    # print original.v_hessian
    # # print result

    # print optimizer._test_necessity_for_finite_difference(1)
    # original.get_v_hessian()
    # print original.v_hessian
    # step = -np.dot(np.linalg.pinv(original.v_hessian), original.v_gradient)
    # original.stepsize = step
    # print step
    # optimizer.add_a_point(original)
    # third_ori = optimizer.update_to_new_point_for_latest_point()
    # print third_ori.ts_state.energy

    # optimizer.start_iterate_optimization(500)
    # latest = optimizer.points[500]
    # print latest.ts_state.coordinates, latest.ts_state.energy
    # print ts_treat.ts_state.coordinates, ts_treat.ts_state.energy
    # print latest.ts_state.coordinates
    # print latest.ts_state.gradient_matrix
    # print latest.ts_state.ic_gradient
    # print ts_treat.ts_state.ic_gradient
    # print optimizer.latest_index
    # print optimizer.points[20].v_gradient
    # optimizer.tweak_hessian_for_latest_point()
    # # print np.linalg.eigh(ts_treat.v_hessian)
    # ts_treat._diagnolize_h_matrix()
    # optimizer.find_stepsize_for_latest_point(method="TRIM")
    # _trial_two = optimizer.update_to_new_point_for_latest_point()
    # print _trial_two.ts_state.energy, _trial_two.ts_state.coordinates
    # print "first point step control",ts_treat.step_control
    # second_point = optimizer.update_to_new_point_for_latest_point()
    # print "second point information", second_point.step_control
    # print "satisfied check" # something need to be fixed
    # print optimizer.verify_new_point_with_point(0, second_point)
    # print ts_treat.step_control
    # new_second_point = optimizer.update_to_new_point_for_latest_point()
    # print "new_second_point"
    # print new_second_point.ts_state.gradient_matrix
    # print np.linalg.norm(new_second_point.ts_state.gradient_matrix)
    # print np.linalg.norm(ts_treat.ts_state.gradient_matrix)
    # optimizer.add_a_point(new_second_point)
    # print "converge", optimizer.verify_convergence_for_latest_point()
    # # print new_second_point.ts_state.energy
    # optimizer.update_trust_radius_latest_point(method="gradient")
    # print new_second_point.step_control
    # optimizer.update_hessian_for_latest_point(method="SR1")
    # optimizer.tweak_hessian_for_latest_point()
    # optimizer.find_stepsize_for_latest_point(method="TRIM")
    # third_trial = optimizer.update_to_new_point_for_latest_point()
    # print optimizer.verify_new_point_with_latest_point(third_trial)
    # print new_second_point.step_control
    # need to update trm. method need to be implelemented here
    ''' step for optimization
    add_a_point
    verify_convergence_for_latest_point
    update_trust_radius_latest_point
    update_hessian_for_latest_point
    tweak_hessian_for_latest_point
    find_stepsize_for_latest_point
    update_to_new_point_for_latest_point
    verify_new_point_with_latest_point

    '''
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
