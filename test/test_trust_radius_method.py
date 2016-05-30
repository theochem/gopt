import numpy as np
from collections import namedtuple
from saddle.tstreat import TS_Treat

def test_trust_radius():
    hessian = np.array([[1,2,3],[2,4,5],[3,5,7]])
    assert np.linalg.det(hessian) != 0
    assert np.allclose(hessian, hessian.T)
    v, w =np.linalg.eigh(hessian)
    new_hessian = np.dot(np.dot(w, np.diag(
        v)), w.T)
    assert np.allclose(new_hessian, hessian)
    print v,"\n", w
    ts = namedtuple("ts_state", "dof")
    ts_sample_0 = ts(dof=3)
    ts_treat_sample = TS_Treat(ts_sample_0, 2)
    # ts_treat_sample.v_hessian = hessian
    # ts_treat_sample.v_gradient = np.array([0.5, 0.1, 0.5])
    # ts_treat_sample.step_control = 0.5
    # result = ts_treat_sample._trust_region_image_potential()
    # print result
    # print np.dot(hessian, result), np.linalg.norm(result)
    # step = ts_treat_sample._step_calculate_max_length(3)
    # print "step",step
test_trust_radius()