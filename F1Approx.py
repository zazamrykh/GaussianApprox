import math
import numpy as np


def stepChange(alpha, grad, grad_before, grad_before_last):
    was_change = False
    for i in range(grad.shape[0]):
        if grad[i] > grad_before[i] and grad_before[i] < grad_before_last[i]:
            alpha[i] = alpha[i] * 0.5
            was_change = True
    return was_change


def gradientDescentF1(X, Y, Z):
    alpha = np.array([0.17, 10000, 0.2, 0.2])
    est = np.array([Z[11], 100, 0., 0.])  # Start estimation
    grad_before_last = est
    grad_before = est
    for i in range(1000):
        grad = gradL(X, Y, Z, est[0], est[1], est[2], est[3])
        if stepChange(alpha, grad, grad_before, grad_before_last):
            grad = gradL(X, Y, Z, est[0], est[1], est[2], est[3])
        # print("Step number " + str(i) + " alpha value: " + str(alpha) + "\n" + str(grad))

        if i % 5000 == 0:
            print(findMSE(X, Y, Z, est[0], est[1], est[2], est[3]))

        est -= alpha * grad
        grad_before_last = grad_before
        grad_before = grad
    return est


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


def findMSE(X, Y, Z, A, sigma, x0, y0):
    squad_error = 0
    N = Z.shape[0]
    for i in range(N):
        squad_error += (Z[i] - F1(A, sigma, x0, y0, X[i], Y[i])) ** 2
    return squad_error / N
