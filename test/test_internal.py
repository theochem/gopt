import numpy as np
from saddle.internal import Internal
import horton as ht

class TestInternal(object):

    @classmethod
    def setup_class(self):
        import horton as ht
        fn_xyz = ht.context.get_fn("test/water.xyz")
        mol = ht.IOData.from_file(fn_xyz)  # create a water molecule
        self.internal = Internal(mol.coordinates, mol.numbers, 0, 1)

    def test_connectivity(self):
        assert np.allclose(self.internal.connectivity, np.eye(3) * -1)
