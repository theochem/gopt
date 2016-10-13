import numpy as np
from saddle.cartesian import Cartesian
import horton as ht


class TestCartesian(object):

    @classmethod
    def setup_class(self):
        import horton as ht
        fn_xyz = ht.context.get_fn("test/water.xyz")
        mol = ht.IOData.from_file(fn_xyz)  # create a water molecule
        self.cartesian = Cartesian(mol.coordinates, mol.numbers, 0, 1)

    def test_coordinates(self):
        ref_coordinates = np.array([
            [0.783837, -0.492236, -0.000000],
            [-0.000000, 0.062020, -0.000000],
            [-0.783837, -0.492236, -0.000000]
        ])
        assert np.allclose(self.cartesian.coordinates /
                           ht.angstrom, ref_coordinates)

    def test_numbers(self):
        ref_numbers = np.array([1, 8, 1])
        assert np.allclose(self.cartesian.numbers, ref_numbers)

    def test_charge_and_spin(self):
        ref_spin = 1
        ref_charge = 0
        assert (self.cartesian.spin == ref_spin)
        assert (self.cartesian.charge == ref_charge)

    def test_distance(self):
        ref_distance = np.linalg.norm(np.array(
            [0.783837, -0.492236, -0.000000]) - np.array(
            [-0.000000, 0.062020, -0.000000]))
        assert self.cartesian.distance(0, 1) / ht.angstrom == ref_distance

    def test_angle(self):
        vector1 = np.array([-0.000000, 0.062020, -0.000000]) - \
            np.array([0.783837, -0.492236, -0.000000])
        vector2 = np.array([-0.000000, 0.062020, -0.000000]) - \
            np.array([-0.783837, -0.492236, -0.000000])
        ref_angle_cos = np.dot(vector1, vector2) / \
            np.linalg.norm(vector1) / np.linalg.norm(vector2)
        assert np.allclose(self.cartesian.angle_cos(0, 1, 2), ref_angle_cos)
        assert np.allclose(
            self.cartesian.angle(0, 1, 2), np.arccos(ref_angle_cos))
