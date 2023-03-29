import numpy as np
from sympy import exp, Symbol

x = Symbol('x')
y = Symbol('y')
A = Symbol('A')
sigma_x = Symbol('sigma_x')
sigma_y = Symbol('sigma_y')
x0 = Symbol('x0')
y0 = Symbol('y0')
F1_func = A * exp(-(x - x0) ** 2 / (2 * sigma_x)) * exp(-(y - y0) ** 2 / (2 * sigma_y)) * \
          (1 + (y - y0) ** 2 / (2 * sigma_y) * exp(-(y - y0) ** 2 / (2 * sigma_y)))
F1_diff_A = F1_func.diff(A)
F1_diff_sigma_x = F1_func.diff(sigma_x)
F1_diff_sigma_y = F1_func.diff(sigma_y)
F1_diff_x0 = F1_func.diff(x0)
F1_diff_y0 = F1_func.diff(y0)


def stepChange(alpha, grad, grad_before, grad_before_last):
    was_change = False
    for i in range(grad.shape[0]):
        if grad[i] > grad_before[i] and grad_before[i] < grad_before_last[i]:
            alpha[i] = alpha[i] * 0.5
            was_change = True
    return was_change


def gradientDescentF3(X, Y, Z):
    alpha = np.array([0.17, 10000, 10000, 0.2, 0.2])
    est = np.array([Z[11], 100, 100, 0., 0.])  # Start estimation
    grad_before_last = est
    grad_before = est
    for i in range(20):
        grad = gradL(X, Y, Z, est[0], est[1], est[2], est[3], est[4])
        if stepChange(alpha, grad, grad_before, grad_before_last):
            grad = gradL(X, Y, Z, est[0], est[1], est[2], est[3], est[4])
        print("Step number " + str(i) + " alpha value: " + str(alpha) + "\n" + str(grad))

        if i % 10 == 0:
            print(findMSE(X, Y, Z, est[0], est[1], est[2], est[3], est[4]))

        est -= alpha * grad
        grad_before_last = grad_before
        grad_before = grad
    return est


def derLA(X, Y, Z, x0_0, y0_0, A_0, sigma_x_0, sigma_y_0):
    result = 0
    for x_i, y_i, z in zip(X, Y, Z):
        result += (F3(x_i, y_i, A_0, sigma_x_0, sigma_y_0, x0_0, y0_0) - z) * \
                  F1_diff_A.evalf(subs={x: x_i, y: y_i, A: A_0, sigma_x: sigma_x_0,
                                        sigma_y: sigma_y_0, x0: x0_0, y0: y0_0})
    result *= 2
    return float(result)


def derLSigmaX(X, Y, Z, x0_0, y0_0, A_0, sigma_x_0, sigma_y_0):
    result = 0
    for x_i, y_i, z in zip(X, Y, Z):
        result += (F3(x_i, y_i, A_0, sigma_x_0, sigma_y_0, x0_0, y0_0) - z) * \
                  F1_diff_sigma_x.evalf(subs={x: x_i, y: y_i, A: A_0, sigma_x: sigma_x_0,
                                              sigma_y: sigma_y_0, x0: x0_0, y0: y0_0})
    result *= 2
    return float(result)


def derLSigmaY(X, Y, Z, x0_0, y0_0, A_0, sigma_x_0, sigma_y_0):
    result = 0
    for x_i, y_i, z in zip(X, Y, Z):
        result += (F3(x_i, y_i, A_0, sigma_x_0, sigma_y_0, x0_0, y0_0) - z) * \
                  F1_diff_sigma_y.evalf(subs={x: x_i, y: y_i, A: A_0, sigma_x: sigma_x_0,
                                              sigma_y: sigma_y_0, x0: x0_0, y0: y0_0})
    result *= 2
    return float(result)


def derLX0(X, Y, Z, x0_0, y0_0, A_0, sigma_x_0, sigma_y_0):
    result = 0
    for x_i, y_i, z in zip(X, Y, Z):
        result += (F3(x_i, y_i, A_0, sigma_x_0, sigma_y_0, x0_0, y0_0) - z) * \
                  F1_diff_x0.evalf(subs={x: x_i, y: y_i, A: A_0, sigma_x: sigma_x_0,
                                              sigma_y: sigma_y_0, x0: x0_0, y0: y0_0})
    result *= 2
    return float(result)


def derLY0(X, Y, Z, x0_0, y0_0, A_0, sigma_x_0, sigma_y_0):
    result = 0
    for x_i, y_i, z in zip(X, Y, Z):
        result += (F3(x_i, y_i, A_0, sigma_x_0, sigma_y_0, x0_0, y0_0) - z) * \
                  F1_diff_y0.evalf(subs={x: x_i, y: y_i, A: A_0, sigma_x: sigma_x_0,
                                         sigma_y: sigma_y_0, x0: x0_0, y0: y0_0})
    result *= 2
    return float(result)


def gradL(X, Y, Z, A_0, sigma_x_0, sigma_y_0, x0_0, y0_0, ):
    return np.array([derLA(X, Y, Z, x0_0, y0_0, A_0, sigma_x_0, sigma_y_0),
                     derLSigmaX(X, Y, Z, x0_0, y0_0, A_0, sigma_x_0, sigma_y_0),
                     derLSigmaY(X, Y, Z, x0_0, y0_0, A_0, sigma_x_0, sigma_y_0),
                     derLX0(X, Y, Z, x0_0, y0_0, A_0, sigma_x_0, sigma_y_0),
                     derLY0(X, Y, Z, x0_0, y0_0, A_0, sigma_x_0, sigma_y_0)], dtype=np.float64)


def F3(x_0, y_0, A_0, sigma_x_0, sigma_y_0, x0_0, y0_0):
    return A_0 * doubleGaussExp(x_0, x0_0, y_0, y0_0, sigma_x_0, sigma_y_0) * (
            1 + (((y_0 - y0_0) ** 2) / (2 * sigma_y_0)) * gaussExp(y_0, y0_0, sigma_y_0))


def gaussExp(x_0, x0_0, sigma_0):
    import math
    return math.exp((-(x_0 - x0_0) ** 2) / (2 * sigma_0))


def doubleGaussExp(x_0, x0_0, y_0, y0_0, sigma_x_0, sigma_y_0):
    return gaussExp(x_0, x0_0, sigma_x_0) * gaussExp(y_0, y0_0, sigma_y_0)


def findMSE(X, Y, Z, A_0, sigma_x_0, sigma_y_0, x0_0, y0_0):
    squad_error = 0
    N = Z.shape[0]
    for i in range(N):
        squad_error += (Z[i] - F3(X[i], Y[i], A_0, sigma_x_0, sigma_y_0, x0_0, y0_0)) ** 2
    return squad_error / N
