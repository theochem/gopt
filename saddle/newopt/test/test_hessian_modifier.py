import numpy as np

from saddle.newopt.hessian_modifier import SaddleHessianModifier


class TestHessianModifier(object):
    @classmethod
    def setup_class(self):
        self.sym_mt = np.array([[2., 5., 9., 16.],
                                [5., 8., 11., 14.],
                                [9., 11., 14., 14.],
                                [16., 14., 14., 21.], ])

    def test_hessian_modifier_all_positive(self):
        w, v = np.linalg.eigh(self.sym_mt)
        answer = SaddleHessianModifier._modify_hessian(
            self.sym_mt, key_ic_number=3, negative_eigen=0)
        ref_mt = np.array([0.005, 0.30280199, 4.04911211, 48.71752423])
        a_w, a_v = np.linalg.eigh(answer)
        assert np.allclose(ref_mt, a_w)
        assert np.allclose(np.square(v), np.square(a_v))

    def test_hessian_modifier_one_negative(self):
        w, v = np.linalg.eigh(self.sym_mt)
        anwser = SaddleHessianModifier._modify_hessian(
            self.sym_mt, negative_eigen=1, key_ic_number=3)
        a_w, a_v = np.linalg.eigh(anwser)
        ref_mt = np.array([-8.06943832, 0.30280199, 4.04911211, 48.71752423])
        assert np.allclose(ref_mt, a_w)
        assert np.allclose(np.square(a_v), np.square(v))

    def test_hessian_modifier_two_negative(self):
        w, v = np.linalg.eigh(self.sym_mt)
        anwser = SaddleHessianModifier._modify_hessian(
            self.sym_mt, negative_eigen=2, key_ic_number=3)
        a_w, a_v = np.linalg.eigh(anwser)
        ref_mt = np.array([-8.06943832, -0.005, 4.04911211, 48.71752423])
        assert np.allclose(ref_mt, a_w)
        assert np.allclose(np.square(a_v), np.square(v))

    def test_hessian_modifier_reduce_to_one_nagetive(self):
        sym_mt = self.sym_mt.copy()
        sym_mt[2, 3] = 17
        sym_mt[3, 2] = 17
        w, v = np.linalg.eigh(sym_mt)
        print w, v
        anwser = SaddleHessianModifier._modify_hessian(
            sym_mt, negative_eigen=1, key_ic_number=3)
        a_w, a_v = np.linalg.eigh(anwser)
        ref_mt = np.array([-0.06679842, 0.005, 2.08628757, 50.75611254])
        assert np.allclose(a_w, ref_mt)
