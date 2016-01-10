import numpy as np
from saddle import *

def test_optimizer_Point():
    test_array = np.array([[ 1.48, -0.93, -0.  ],
                           [-0.  ,  0.11, -0.  ],
                           [-1.48, -0.93, -0.  ]])
    test_point = Point(test_array)
    other_array = np.arange(9.).reshape(3,3)
    other_point = Point(other_array)
    sample =  np.array([[1.48, 0.07, 2.],
                        [3.  , 4.11, 5.],
                        [4.52, 6.07, 8.]])
    assert np.allclose((test_point + other_point).coordinates, sample)

    sample = np.array([[0, .5, 1.],[1.5, 2., 2.5],[3., 3.5, 4.]])
    assert np.allclose((other_point / 2).coordinates, sample)