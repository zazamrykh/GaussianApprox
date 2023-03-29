import math
import numpy as np


def stepChange(alpha, grad, grad_before, grad_before_last):
    was_change = False
    for i in range(grad.shape[0]):
        if grad[i] > grad_before[i] and grad_before[i] < grad_before_last[i]:
            alpha[i] = alpha[i] * 0.5
            was_change = True
    return was_change


def gradientDescentF2(X, Y, Z):
    alpha = np.array([0.17, 10000, 10000, 0.2, 0.2])
    est = np.array([Z[11], 100, 100, 0., 0.])  # Start estimation
    grad_before_last = est
    grad_before = est
    for i in range(5001):
        grad = gradL(X, Y, Z, est[0], est[1], est[2], est[3], est[4])
        if stepChange(alpha, grad, grad_before, grad_before_last):
            grad = gradL(X, Y, Z, est[0], est[1], est[2], est[3], est[4])

        if i % 5000 == 0:
            print(findMSE(X, Y, Z, est[0], est[1], est[2], est[3], est[4]))

        est -= alpha * grad
        grad_before_last = grad_before
        grad_before = grad
    return est


def gradL(X, Y, Z, A, sigma_x, sigma_y, x0, y0):
    return np.array([derLA(X, Y, Z, x0, y0, A, sigma_x, sigma_y),
                     derLSigmaX(X, Y, Z, x0, y0, A, sigma_x, sigma_y),
                     derLSigmaY(X, Y, Z, x0, y0, A, sigma_x, sigma_y),
                     derLX0(X, Y, Z, x0, y0, A, sigma_x, sigma_y),
                     derLY0(X, Y, Z, x0, y0, A, sigma_x, sigma_y)], dtype=np.float64)


def F2(A, sigma_x, sigma_y, x0, y0, x, y):
    return A * doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y)


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


def findMSE(X, Y, Z, A, sigma_x, sigma_y, x0, y0):
    squad_error = 0
    N = Z.shape[0]
    for i in range(N):
        squad_error += (Z[i] - F2(A, sigma_x, sigma_y, x0, y0, X[i], Y[i])) ** 2
    return squad_error / N
