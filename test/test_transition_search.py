import horton as ht
import numpy as np
from saddle.TransitionSearch import *
from saddle.tstreat import TS_Treat
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
    assert np.allclose(ts_treat.v_matrix, test_v)
    print test_v.shape
    print reactant.natom

if __name__ == '__main__':
    test_transitionsearch_cl_h_br()