import math
import numpy as np

from FApprox import FApprox


def gaussExp(x, x0, sigma):
    return math.exp((-(x - x0) ** 2) / (2 * sigma))


def doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y):
    return gaussExp(x, x0, sigma_x) * gaussExp(y, y0, sigma_y)


def F3(A, sigma_x, sigma_y, x0, y0, x, y):
    return A * doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y) * (
            1 + (((y - y0) ** 2) / (2 * sigma_y)) * gaussExp(y, y0, sigma_y))


def gradL(X, Y, Z, A, sigma_x, sigma_y, x0, y0):
    return np.array([derLA(X, Y, Z, x0, y0, A, sigma_x, sigma_y),
                     derLSigmaX(X, Y, Z, x0, y0, A, sigma_x, sigma_y),
                     derLSigmaY(X, Y, Z, x0, y0, A, sigma_x, sigma_y),
                     derLX0(X, Y, Z, x0, y0, A, sigma_x, sigma_y),
                     derLY0(X, Y, Z, x0, y0, A, sigma_x, sigma_y)], dtype=np.float64)


class F3ApproxClass(FApprox):
    def __init__(self, number_of_repeat, start_est, start_alpha, step_change_coef=0.9):
        super().__init__(number_of_repeat, start_est, start_alpha, F3, gradL, step_change_coef)


def derLA(X, Y, Z, x0, y0, A, sigma_x, sigma_y):
    result = 0
    for x, y, z in zip(X, Y, Z):
        result += (F3(A, sigma_x, sigma_y, x0, y0, x, y) - z) * derF3A(x, y, x0, y0, sigma_x, sigma_y)
    result *= 2
    return result


def derLX0(X, Y, Z, x0, y0, A, sigma_x, sigma_y):
    result = 0
    for x, y, z in zip(X, Y, Z):
        result += (F3(A, sigma_x, sigma_y, x0, y0, x, y) - z) * derF3X0(x, y, x0, y0, A, sigma_x, sigma_y)
    result *= 2
    return result


def derLY0(X, Y, Z, x0, y0, A, sigma_x, sigma_y):
    result = 0
    for x, y, z in zip(X, Y, Z):
        result += (F3(A, sigma_x, sigma_y, x0, y0, x, y) - z) * derF3Y0(x, y, x0, y0, A, sigma_x, sigma_y)
    result *= 2
    return result


def derLSigmaX(X, Y, Z, x0, y0, A, sigma_x, sigma_y):
    result = 0
    for x, y, z in zip(X, Y, Z):
        result += (F3(A, sigma_x, sigma_y, x0, y0, x, y) - z) * derF3SigmaX(x, y, x0, y0, A, sigma_x, sigma_y)
    result *= 2
    return result


def derLSigmaY(X, Y, Z, x0, y0, A, sigma_x, sigma_y):
    result = 0
    for x, y, z in zip(X, Y, Z):
        result += (F3(A, sigma_x, sigma_y, x0, y0, x, y) - z) * derF3SigmaY(x, y, x0, y0, A, sigma_x, sigma_y)
    result *= 2
    return result


def derF3X0(x, y, x0, y0, A, sigma_x, sigma_y):
    return A * doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y) * (x - x0) / sigma_x


def derF3A(x, y, x0, y0, sigma_x, sigma_y):
    return (
            doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y) +
            gaussExp(y, y0, sigma_y / 2) * ((y - y0) ** 2 / (2 * sigma_y))
    )


def derF3Y0(x, y, x0, y0, A, sigma_x, sigma_y):
    return (
            A * doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y) * (y - y0) / sigma_y +
            (A / sigma_y) * (y - y0) * gaussExp(y, y0, sigma_y / 2) *
            (((y - y0) ** 2 / sigma_y) - 1)
    )


def derF3SigmaY(x, y, x0, y0, A, sigma_x, sigma_y):
    return (A * doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y) * ((y - y0) ** 2 * 2 / ((2 * sigma_y) ** 2)) + (
            A / 2) * (y - y0) ** 2 *
            gaussExp(y, y0, sigma_y / 2) * ((y - y0) ** 2 * (sigma_x ** (-3)) - (sigma_y ** (-2))))


def derF3SigmaX(x, y, x0, y0, A, sigma_x, sigma_y):
    return (
            A * doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y) * ((x - x0) ** 2 * 2 / ((2 * sigma_x) ** 2))
    )
