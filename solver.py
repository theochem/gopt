import numpy as np


def ridders_solver(func, x1, x2, iteration=30, error=10e-6):

    """The ridders solver to solver nonlinear equation to find a mathematical
    root for a continuous function. the value of the two end should be of
    different sign.

    Args:
        func (function): function to find the right root
        x1 (float): left end of interval
        x2 (float): right end of interval
        iteration (int, optional): numbers of iterations, default is 30
        error (float, optional): the threshold for convergence, default is 10e-6

    Raises:
        PositiveProductError: when the function value of two ends of the interval
                              is of the same sign
    return:
        (float): the root for function in the interval between x1 and x2


    """
    f1 = func(x1)
    f2 = func(x2)
    if f1 * f2 > 0:
        raise PositiveProductError("The two end point are of same sign")
    answer = 0
    if np.allclose(f1, 0):
        return x1
    elif np.allclose(f2, 0):
        return x2
    for i in range(iteration):
        x3 = 0.5 * (x1 + x2)
        f3 = func(x3)
        s = np.sqrt(f3 * f3 - f1 * f2)
        if np.allclose(s, 0):
            return answer
        x4 = x3 + (x3 - x1) * np.sign(f1 - f2) * f3 / s
        if abs(x4 - answer) < error:
            return answer
        answer = x4
        f4 = func(x4)
        if np.allclose(f4, 0.):
            return answer
        if np.sign(f4) != np.sign(f3):
            x1, f1 = x3, f3
            x2, f2 = x4, f4
        elif np.sign(f4) != np.sign(f1):
            x2, f2 = x4, f4
        elif np.sign(f4) != np.sign(f2):
            x1, f1 = x4, f4
        else:
            raise PositiveProductError("The two end point are of same sign")



class PositiveProductError(Exception):
    pass

if __name__ == "__main__":
    fcn = lambda x: x**2 - 4
    result = ridders_solver(fcn, -10, -1)
    print(result)
