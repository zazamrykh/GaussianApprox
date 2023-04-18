import math
import numpy as np

from FApprox import FApprox


class F1ApproxClass(FApprox):
    def __init__(self, number_of_repeat, start_est, start_alpha, step_change_coef=0.8):
        super().__init__(number_of_repeat, start_est, start_alpha, F1, gradL, step_change_coef)


def gradL(X, Y, Z, A, sigma, x0, y0):
    return np.array([derLA(X, Y, Z, x0, y0, A, sigma),
                     derLSigma(X, Y, Z, x0, y0, A, sigma),
                     derLX0(X, Y, Z, x0, y0, A, sigma),
                     derLY0(X, Y, Z, x0, y0, A, sigma)], dtype=np.float64)


def F1(A, sigma, x0, y0, x, y):
    return A * doubleGaussExp(x, x0, y, y0, sigma)


def gaussExp(x, x0, sigma):
    return math.exp((-(x - x0) ** 2) / (2 * sigma))


def doubleGaussExp(x, x0, y, y0, sigma):
    return gaussExp(x, x0, sigma) * gaussExp(y, y0, sigma)


def derF1A(x, y, x0, y0, sigma):
    return doubleGaussExp(x, x0, y, y0, sigma)


def derF1X0(x, y, x0, y0, A, sigma):
    return A * doubleGaussExp(x, x0, y, y0, sigma) * (x - x0) / sigma


def derF1Y0(x, y, x0, y0, A, sigma):
    return A * doubleGaussExp(x, x0, y, y0, sigma) * (y - y0) / sigma


def derF1Sigma(x, y, x0, y0, A, sigma):
    return A * doubleGaussExp(x, x0, y, y0, sigma) * ((x - x0) ** 2 + (y - y0) ** 2) * 2 / ((2 * sigma) ** 2)


def derLA(X, Y, Z, x0, y0, A, sigma):
    result = 0
    for x, y, z in zip(X, Y, Z):
        result += (F1(A, sigma, x0, y0, x, y) - z) * derF1A(x, y, x0, y0, sigma)
    result *= 2
    return result


def derLX0(X, Y, Z, x0, y0, A, sigma):
    result = 0
    for x, y, z in zip(X, Y, Z):
        result += (F1(A, sigma, x0, y0, x, y) - z) * derF1X0(x, y, x0, y0, A, sigma)
    result *= 2
    return result


def derLY0(X, Y, Z, x0, y0, A, sigma):
    result = 0
    for x, y, z in zip(X, Y, Z):
        result += (F1(A, sigma, x0, y0, x, y) - z) * derF1Y0(x, y, x0, y0, A, sigma)
    result *= 2
    return result


def derLSigma(X, Y, Z, x0, y0, A, sigma):
    result = 0
    for x, y, z in zip(X, Y, Z):
        result += (F1(A, sigma, x0, y0, x, y) - z) * derF1Sigma(x, y, x0, y0, A, sigma)
    result *= 2
    return result
