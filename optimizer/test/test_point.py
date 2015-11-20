import numpy as np
from saddle import *

def test_optimizer_Point():
    test_array = np.array([[ 1.48, -0.93, -0.  ],
                           [-0.  ,  0.11, -0.  ],
                           [-1.48, -0.93, -0.  ]])
    test_point = Point(test_array)
    other_array = np.arange(9.).reshape(3,3)
    other_point = Point(other_array)
    diff = (test_point + other_point).coordinates - np.array([[1.48, 0.07, 2.],
                                                              [3.  , 4.11, 5.],
                                                              [4.52, 6.07, 8.]])
    diff = diff.flatten()
    assert np.dot(diff,diff) < 10e-15

    diff = (other_point / 2).coordinates - np.array([[0, .5, 1.],[1.5, 2., 2.5],[3., 3.5, 4.]])
    diff = diff.flatten()
    assert np.dot(diff,diff) < 10e-15