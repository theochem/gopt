from __future__ import absolute_import

import numpy as np

from .abclass import HessianModifier

__all__ = ('SaddleHessianModifier', )


class SaddleHessianModifier(HessianModifier):

    positive = 0.005
    negative = -0.005

    @staticmethod
    def modify_hessian(point, key_ic_number=0, negative_eigen=0):
        new_hessian = SaddleHessianModifier._modify_hessian(
            point.hessian, key_ic_number, negative_eigen)
        point.set_hessian(new_hessian)

    @staticmethod
    def _modify_hessian(hessian, key_ic_number,
                        negative_eigen):  # need to be tested
        assert hessian.shape[0] == hessian.shape[1]
        assert len(hessian.shape) == 2
        dimension = hessian.shape[0]
        assert key_ic_number <= dimension
        hessian = hessian.copy()
        non_reduce = hessian[key_ic_number:, key_ic_number:]
        #result = SaddleHessianModifier._change_to_positive_eigen(non_reduce)
        if non_reduce.size == 0:
            hessian[
                key_ic_number:,
                key_ic_number:] = SaddleHessianModifier._change_to_positive_eigen(
                    non_reduce)
        new_hessian = SaddleHessianModifier._select_main_column(
            hessian, key_ic_number, negative_eigen)
        return new_hessian

    @staticmethod
    def _select_main_column(hessian, key_ic_number, negative_eigen):
        w, v = np.linalg.eigh(hessian)
        total_neg = np.sum([w < 0.])
        if total_neg == negative_eigen:
            neg_index = np.arange(total_neg)
            pos_index = np.arange(total_neg, len(w))
        elif total_neg > negative_eigen:  # have too many negative eigenvalue
            diff = total_neg - negative_eigen
            fraction = np.sum(np.square(v[:key_ic_number, :total_neg]), axis=0)
            neg_index = fraction.argsort()[::-1][:negative_eigen]
            pos_index = np.append(fraction.argsort()[::-1][negative_eigen:],
                                  np.arange(total_neg, len(w)))
        else:
            diff = negative_eigen - total_neg
            fraction = np.sum(np.square(v[:key_ic_number, total_neg:]), axis=0)
            pos_index = (fraction.argsort() + total_neg)[:-diff]
            neg_index = np.append((fraction.argsort() + total_neg)[-diff:],
                                  np.arange(total_neg))
        ngt_e_vl = w[neg_index]  # negative eigenvalues
        pst_e_vl = w[pos_index]  # positive eigenvalues
        ngt_e_vl[
            ngt_e_vl > SaddleHessianModifier.
            negative] = SaddleHessianModifier.negative  # change the nagetive eigenvalues to <= -threshold
        pst_e_vl[
            pst_e_vl < SaddleHessianModifier.
            positive] = SaddleHessianModifier.positive  # change the positive eigenvalues to >= threshold
        w[neg_index] = ngt_e_vl
        w[pos_index] = pst_e_vl
        new_hessian = np.dot(v, np.dot(np.diag(w), v.T))
        return new_hessian

    @staticmethod
    def _change_to_positive_eigen(hessian, threshold=0.):
        w, v = np.linalg.eigh(hessian)
        w[w < threshold] = 0.001
        result = np.dot(np.dot(v, np.diag(w)), v.T)
        return result
