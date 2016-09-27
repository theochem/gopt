import numpy as np
from saddle.internal import Internal
from saddle.opt import Point
from copy import deepcopy
import horton as ht

class TestInternal(object):

    @classmethod
    def setup_class(self):
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
        # print mol._cc_to_ic_hessian[2]
        assert np.allclose(mol._cc_to_ic_hessian[2], np.array([
            [ 0.30385023,  0.1432367 ,  0.        , -0.27008904,
            -0.19098226,-0.        , -0.03376119,  0.04774557,  0.        ],
            [ 0.1432367 , -0.20256656,  0.        , -0.09549113,
            0.13504407,-0.        , -0.04774557,  0.06752248,  0.        ],
            [ 0.        ,  0.        ,  0.10128367, -0.        , -0.        ,
            -0.40513401,  0.        ,  0.        ,  0.30385034],
            [-0.27008904, -0.09549113, -0.        ,  0.54017808,  0.,
            0.        , -0.27008904,  0.09549113, -0.        ],
            [-0.19098226,  0.13504407, -0.        ,  0.        ,
            -0.27008815,0.        ,  0.19098226,  0.13504407, -0.        ],
            [-0.        , -0.        , -0.40513401,  0.        ,  0.,
            0.81026801, -0.        , -0.        , -0.40513401],
            [-0.03376119, -0.04774557,  0.        , -0.27008904,
            0.19098226,-0.        ,  0.30385023, -0.1432367 ,  0.        ],
            [ 0.04774557,  0.06752248,  0.        ,  0.09549113,
            0.13504407,-0.        , -0.1432367 , -0.20256656,  0.        ],
            [ 0.        ,  0.        ,  0.30385034, -0.        , -0.,
            -0.40513401,  0.        ,  0.        ,  0.10128367]]))
        mol.set_target_ic((1.6,1.7,-0.5))
        assert np.allclose(mol.target_ic, np.array([1.6, 1.7, -0.5]))

    def test_dihedral_add(self):
        fn_xyz = ht.context.get_fn("test/2h-azirine.xyz")
        mol = ht.IOData.from_file(fn_xyz)  # create a water molecule
        internal = Internal(mol.coordinates, mol.numbers, 0, 1)
        internal.add_bond(0, 1)
        internal.add_bond(1, 2)
        internal.add_bond(1, 3)
        internal.add_dihedral(0, 2, 3, 4)
        assert len(internal.ic) == 3
        internal.add_dihedral(0, 1, 2, 3)
        assert len(internal.ic) == 4
        ref_hessian = np.array([
            [  7.21126973e-18,  -6.65055700e-18,   1.19079554e-01,
             -4.42043147e-17,   5.19145620e-17,  -1.72735840e-01,
              9.48268842e-18,  -7.16890755e-18,   5.36562861e-02,
              2.75103565e-17,  -3.80950975e-17,  -6.73656683e-18],
            [ -6.65055700e-18,   5.51999042e-18,  -1.64895981e-01,
              1.78057937e-17,  -2.45419114e-17,  -4.76433296e-02,
              9.51393806e-19,   2.25720657e-18,   2.12539310e-01,
             -1.21066305e-17,   1.67647144e-17,   2.96459719e-18],
            [  1.19079554e-01,  -1.64895981e-01,   2.91594685e-02,
             -7.64902140e-03,   1.05920189e-02,  -1.60482517e-02,
             -9.03966477e-02,   1.25177189e-01,  -1.82618651e-02,
             -2.10338849e-02,   2.91267724e-02,   5.15064830e-03],
            [ -4.37560341e-17,   1.79559294e-17,  -7.64902140e-03,
             -1.77407422e-01,  -3.90537615e-02,   4.85047601e-01,
              1.41837266e-01,  -1.18599480e-01,  -2.38868527e-01,
              3.55701553e-02,   1.57653241e-01,  -2.38530052e-01],
            [  5.23821935e-17,  -2.39950522e-17,   1.05920189e-02,
             -3.90537615e-02,   4.48346617e-01,  -5.13806199e-01,
              8.83097038e-02,  -2.30035539e-01,   1.72908552e-01,
             -4.92559423e-02,  -2.18311078e-01,   3.30305628e-01],
            [ -1.72735840e-01,  -4.76433296e-02,  -1.60482517e-02,
              4.85047601e-01,  -5.13806199e-01,  -1.78286005e-01,
             -5.61974695e-02,   2.57460690e-01,   7.53416936e-02,
             -2.56114291e-01,   3.03988839e-01,   1.18992563e-01],
            [  9.03440782e-18,   8.01258132e-19,  -9.03966477e-02,
              1.41837266e-01,   8.83097038e-02,  -5.61974695e-02,
             -6.07493766e-02,   6.31263778e-03,   8.14067841e-02,
             -8.10878896e-02,  -9.46223416e-02,   6.51873331e-02],
            [ -7.63653899e-18,   1.71034741e-18,   1.25177189e-01,
             -1.18599480e-01,  -2.30035539e-01,   2.57460690e-01,
              6.31263778e-03,   9.90068000e-02,  -2.92369408e-01,
              1.12286842e-01,   1.31028739e-01,  -9.02684706e-02],
            [  5.36562861e-02,   2.12539310e-01,  -1.82618651e-02,
             -2.38868527e-01,   1.72908552e-01,   7.53416936e-02,
              8.14067841e-02,  -2.92369408e-01,  -2.05602033e-02,
              1.03805457e-01,  -9.30784543e-02,  -3.65196252e-02],
            [  2.75103565e-17,  -1.21066305e-17,  -2.10338849e-02,
              3.55701553e-02,  -4.92559423e-02,  -2.56114291e-01,
             -8.10878896e-02,   1.12286842e-01,   1.03805457e-01,
              4.55177343e-02,  -6.30308998e-02,   1.73342719e-01],
            [ -3.80950975e-17,   1.67647144e-17,   2.91267724e-02,
              1.57653241e-01,  -2.18311078e-01,   3.03988839e-01,
             -9.46223416e-02,   1.31028739e-01,  -9.30784543e-02,
             -6.30308998e-02,   8.72823392e-02,  -2.40037158e-01],
            [ -6.73656683e-18,   2.96459719e-18,   5.15064830e-03,
             -2.38530052e-01,   3.30305628e-01,   1.18992563e-01,
              6.51873331e-02,  -9.02684706e-02,  -3.65196252e-02,
              1.73342719e-01,  -2.40037158e-01,  -8.76235860e-02]])
        assert np.allclose(internal._cc_to_ic_hessian[3][:12,:12], ref_hessian)

    def test_cost_function(self):
        mol = deepcopy(self.internal)
        mol.add_bond(0, 1)
        mol.add_bond(1, 2)
        mol.add_angle_cos(0, 1, 2)
        assert np.allclose(mol.ic_values, [1.8141372422079882, 1.8141372422079882, -0.33333406792305265])
        mol.set_target_ic([1.7, 1.7, -0.4])
        assert np.allclose(mol.target_ic, [1.7, 1.7, -0.4])
        v, d, dd = mol.cost_value
        assert np.allclose(0.030498966617378116, v)
        ref_gradient = np.array([0.22827448441597653, 0.22827448441597653, 0.13333186415389475])
        assert np.allclose(d, ref_gradient)
        ref_hessian = np.array([[2., 0., 0.,],
                                [0., 2., 0.,],
                                [0., 0., 2.,]])
        assert np.allclose(dd, ref_hessian)
        # assert False
        new_v, xd, xdd = mol.cost_value_in_cc
        assert new_v == v
        assert np.allclose(xd, np.dot(mol._cc_to_ic_gradient.T, d))
        ref_x_hessian = np.dot(np.dot(mol._cc_to_ic_gradient.T, dd), mol._cc_to_ic_gradient)
        K = np.tensordot(d, mol._cc_to_ic_hessian, 1)
        ref_x_hessian += K
        assert np.allclose(xdd, ref_x_hessian)
        new_coor = np.array([
            [ 1.40, -0.93019123, -0.        ],
            [-0.        ,  0.11720081, -0.        ],
            [-1.40, -0.93019123, -0.        ]])
        mol.set_new_coordinates(new_coor)
        assert np.allclose(mol.ic_values, [1.7484364736491811, 1.7484364736491811, -0.28229028459335431])
        assert np.allclose(mol._cc_to_ic_gradient[0, :6], np.array([ 0.80071539, -0.59904495,  0.  ,-0.80071539, 0.59904495, -0.]))
        ref_hessian = np.array([
            [ 0.20524329,  0.27433912,  0.        , -0.20524329, -0.27433912,
            -0.        ],
            [ 0.27433912,  0.36669628,  0.        , -0.27433912, -0.36669628,
            -0.        ],
            [ 0.        ,  0.        ,  0.57193957, -0.        , -0.        ,
            -0.57193957],
            [-0.20524329, -0.27433912, -0.        ,  0.20524329,  0.27433912,
            0.        ],
            [-0.27433912, -0.36669628, -0.        ,  0.27433912,  0.36669628,
            0.        ],
            [-0.        , -0.        , -0.57193957,  0.        ,  0.        ,
            0.57193957]])
        assert np.allclose(mol._cc_to_ic_hessian[0, :6, :6], ref_hessian)

    def test_transform_function(self):
        mol = deepcopy(self.internal)
        mol.add_bond(0, 1)
        mol.add_bond(1, 2)
        mol.add_angle_cos(0, 1, 2)
        mol.set_target_ic([1.7, 1.7, -0.4])
        n_p = mol._create_geo_point()
        assert isinstance(n_p, Point)
        assert n_p.trust_radius == 1.7320508075688772
        result = mol.converge_to_target_ic()
        g_array = result.cost_value_in_cc[1]
        assert len(g_array[abs(g_array) > 3e-6]) == 0
