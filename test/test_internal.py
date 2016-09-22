import numpy as np
from saddle.internal import Internal
from copy import deepcopy


class TestInternal(object):

    @classmethod
    def setup_class(self):
        import horton as ht
        fn_xyz = ht.context.get_fn("test/water.xyz")
        mol = ht.IOData.from_file(fn_xyz)  # create a water molecule
        self.internal = Internal(mol.coordinates, mol.numbers, 0, 1)

    def test_connectivity(self):
        mol = deepcopy(self.internal)
        assert np.allclose(mol.connectivity, np.eye(3) * -1)

    def test_add_bond(self):
        mol = deepcopy(self.internal)
        init_con = np.eye(3) * -1
        assert np.allclose(mol.connectivity, init_con)
        mol.add_bond(1, 0)
        new_con = init_con.copy()
        new_con[0, 1] = 1
        new_con[1, 0] = 1
        assert np.allclose(mol.connectivity, new_con)
        assert np.allclose(mol.ic_values, np.array([1.81413724]))
        assert mol.ic[0].atoms == (0, 1)
        mol.add_bond(0, 1)
        ref_hessian = np.array([
            [[0.18374187, 0.25985046, 0., -0.18374187, -0.25985046, 0., 0., 0., 0.],
             [0.25985046, 0.36748434, 0., -0.25985046, -0.36748434, 0., 0., 0., 0.],
                [0., 0., 0.55122621, 0., 0., -0.55122621, 0., 0., 0.],
                [-0.18374187, -0.25985046, 0., 0.18374187,
                    0.25985046, 0., 0., 0., 0.],
                [-0.25985046, -0.36748434, 0., 0.25985046,
                    0.36748434, 0., 0., 0., 0.],
                [0., 0., -0.55122621, 0., 0., 0.55122621, 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0.]]
        ])
        assert np.allclose(mol._cc_to_ic_hessian, ref_hessian)
        assert len(mol.ic) == 1
        assert np.allclose(mol.connectivity, new_con)
        mol.add_bond(2, 1)
        assert len(mol.ic) == 2
        assert mol.ic[1].atoms == (1, 2)
        assert np.allclose(mol._cc_to_ic_gradient, np.array(
            [[0.81649681, -0.57734995,  0.,     -0.81649681,  0.57734995, 0., 0., 0., 0.], [0., 0., 0., 0.81649681, 0.57734995, 0., -0.81649681, -0.57734995, 0.]]))

    def test_angle_add(self):
        mol = deepcopy(self.internal)
        mol.add_angle_cos(0, 1, 2)
        assert len(mol.ic) == 0
        mol.add_bond(0, 1)
        mol.add_bond(1, 2)
        mol.add_angle_cos(0, 1, 2)
        assert len(mol.ic) == 3
        assert np.allclose(mol.ic[2].value, -0.33333406792305265)
        assert np.allclose(mol._cc_to_ic_gradient[2], np.array([-0.3000493, -0.42433414,  0., 0.,  0.84866827, -0., 0.3000493, -0.42433414,  0.]))
        print mol._cc_to_ic_hessian[2]
        assert np.allclose(mol._cc_to_ic_hessian[2], np.array([[ 0.30385023,  0.1432367 ,  0.        , -0.27008904, -0.19098226,-0.        , -0.03376119,  0.04774557,  0.        ],
       [ 0.1432367 , -0.20256656,  0.        , -0.09549113,  0.13504407,-0.        , -0.04774557,  0.06752248,  0.        ],
       [ 0.        ,  0.        ,  0.10128367, -0.        , -0.        ,-0.40513401,  0.        ,  0.        ,  0.30385034],
       [-0.27008904, -0.09549113, -0.        ,  0.54017808,  0.        ,0.        , -0.27008904,  0.09549113, -0.        ],
       [-0.19098226,  0.13504407, -0.        ,  0.        , -0.27008815,0.        ,  0.19098226,  0.13504407, -0.        ],
       [-0.        , -0.        , -0.40513401,  0.        ,  0.        ,0.81026801, -0.        , -0.        , -0.40513401],
       [-0.03376119, -0.04774557,  0.        , -0.27008904,  0.19098226,-0.        ,  0.30385023, -0.1432367 ,  0.        ],
       [ 0.04774557,  0.06752248,  0.        ,  0.09549113,  0.13504407,-0.        , -0.1432367 , -0.20256656,  0.        ],
       [ 0.        ,  0.        ,  0.30385034, -0.        , -0.        ,-0.40513401,  0.        ,  0.        ,  0.10128367]]))
