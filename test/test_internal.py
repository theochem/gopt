import numpy as np
from saddle.internal import Internal


class TestInternal(object):

    @classmethod
    def setup_class(self):
        import horton as ht
        fn_xyz = ht.context.get_fn("test/water.xyz")
        mol = ht.IOData.from_file(fn_xyz)  # create a water molecule
        self.internal = Internal(mol.coordinates, mol.numbers, 0, 1)

    def test_add_bond(self):
        init_con = np.eye(3) * -1
        assert np.allclose(self.internal.connectivity, init_con)
        self.internal.add_bond(1, 0)
        new_con = init_con.copy()
        new_con[0, 1] = 1
        new_con[1, 0] = 1
        assert np.allclose(self.internal.connectivity, new_con)
        assert np.allclose(self.internal.ic_values, np.array([1.81413724]))
        assert self.internal.ic[0].atoms == (0, 1)
        self.internal.add_bond(0, 1)
        ref_hessian = np.array([
        [[0.18374187, 0.25985046, 0., -0.18374187, -0.25985046, 0., 0., 0., 0.],
          [0.25985046, 0.36748434, 0., -0.25985046, -0.36748434, 0., 0., 0., 0.],
          [0., 0., 0.55122621, 0., 0., -0.55122621, 0., 0., 0.],
          [-0.18374187, -0.25985046, 0., 0.18374187, 0.25985046, 0., 0., 0., 0.],
          [-0.25985046, -0.36748434, 0., 0.25985046, 0.36748434, 0., 0., 0., 0.],
          [0., 0., -0.55122621, 0., 0., 0.55122621, 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.]]
        ])
        assert np.allclose(self.internal._cc_to_ic_hessian, ref_hessian)
        assert len(self.internal.ic) == 1
        assert np.allclose(self.internal.connectivity, new_con)
        self.internal.add_bond(2, 1)
        assert len(self.internal.ic) == 2
        assert self.internal.ic[1].atoms == (1, 2)
        assert np.allclose(self.internal._cc_to_ic_gradient, np.array(
            [[0.81649681, -0.57734995,  0.,     -0.81649681,  0.57734995, 0., 0., 0., 0.], [0., 0., 0., 0.81649681, 0.57734995, 0., -0.81649681, -0.57734995, 0.]]))

        self.internal.print_connectivity()
        assert False
