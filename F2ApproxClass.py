import math
import numpy as np

from FApprox import FApprox


def F2(A, sigma_x, sigma_y, x0, y0, x, y):
    return A * doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y)


class F2ApproxClass(FApprox):
    def __init__(self, number_of_repeat, start_est, start_alpha):
        super().__init__(number_of_repeat, start_est, start_alpha, F2, gradL)


def gradL(X, Y, Z, A, sigma_x, sigma_y, x0, y0):
    return np.array([derLA(X, Y, Z, x0, y0, A, sigma_x, sigma_y),
                     derLSigmaX(X, Y, Z, x0, y0, A, sigma_x, sigma_y),
                     derLSigmaY(X, Y, Z, x0, y0, A, sigma_x, sigma_y),
                     derLX0(X, Y, Z, x0, y0, A, sigma_x, sigma_y),
                     derLY0(X, Y, Z, x0, y0, A, sigma_x, sigma_y)], dtype=np.float64)


def gaussExp(x, x0, sigma):
    return math.exp((-(x - x0) ** 2) / (2 * sigma))


def doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y):
    return gaussExp(x, x0, sigma_x) * gaussExp(y, y0, sigma_y)


def derF2A(x, y, x0, y0, sigma_x, sigma_y):
    return doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y)


def derF2X0(x, y, x0, y0, A, sigma_x, sigma_y):
    return A * doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y) * (x - x0) / sigma_x


def derF2Y0(x, y, x0, y0, A, sigma_x, sigma_y):
    return A * doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y) * (y - y0) / sigma_y


def derF2SigmaX(x, y, x0, y0, A, sigma_x, sigma_y):
    return A * doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y) * ((x - x0) ** 2 * 2 / ((2 * sigma_x) ** 2))


def derF2SigmaY(x, y, x0, y0, A, sigma_x, sigma_y):
    return A * doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y) * ((y - y0) ** 2 * 2 / ((2 * sigma_y) ** 2))


def derLA(X, Y, Z, x0, y0, A, sigma_x, sigma_y):
    result = 0
    for x, y, z in zip(X, Y, Z):
        result += (F2(A, sigma_x, sigma_y, x0, y0, x, y) - z) * derF2A(x, y, x0, y0, sigma_x, sigma_y)
    result *= 2
    return result


def derLX0(X, Y, Z, x0, y0, A, sigma_x, sigma_y):
    result = 0
    for x, y, z in zip(X, Y, Z):
        result += (F2(A, sigma_x, sigma_y, x0, y0, x, y) - z) * derF2X0(x, y, x0, y0, A, sigma_x, sigma_y)
    result *= 2
    return result


def derLY0(X, Y, Z, x0, y0, A, sigma_x, sigma_y):
    result = 0
    for x, y, z in zip(X, Y, Z):
        result += (F2(A, sigma_x, sigma_y, x0, y0, x, y) - z) * derF2Y0(x, y, x0, y0, A, sigma_x, sigma_y)
    result *= 2
    return result


def derLSigmaX(X, Y, Z, x0, y0, A, sigma_x, sigma_y):
    result = 0
    for x, y, z in zip(X, Y, Z):
        result += (F2(A, sigma_x, sigma_y, x0, y0, x, y) - z) * derF2SigmaX(x, y, x0, y0, A, sigma_x, sigma_y)
    result *= 2
    return result


def derLSigmaY(X, Y, Z, x0, y0, A, sigma_x, sigma_y):
    result = 0
    for x, y, z in zip(X, Y, Z):
        result += (F2(A, sigma_x, sigma_y, x0, y0, x, y) - z) * derF2SigmaY(x, y, x0, y0, A, sigma_x, sigma_y)
    result *= 2
    return result
