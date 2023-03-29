import numpy as np
from sympy import exp, Symbol

x = Symbol('x')
y = Symbol('y')
A = Symbol('A')
sigma = Symbol('sigma')
x0 = Symbol('x0')
y0 = Symbol('y0')
F1_func = A * exp(-(x - x0) ** 2 / (2 * sigma)) * exp(-(y - y0) ** 2 / (2 * sigma))
F1_diff_A = F1_func.diff(A)
F1_diff_sigma = F1_func.diff(sigma)
F1_diff_x0 = F1_func.diff(x0)
F1_diff_y0 = F1_func.diff(y0)


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
    for i in range(20):
        grad = gradL(X, Y, Z, est[0], est[1], est[2], est[3])
        if stepChange(alpha, grad, grad_before, grad_before_last):
            grad = gradL(X, Y, Z, est[0], est[1], est[2], est[3])
        print("Step number " + str(i) + " alpha value: " + str(alpha) + "\n" + str(grad))

        if i % 10 == 0:
            print(findMSE(X, Y, Z, est[0], est[1], est[2], est[3]))

        est -= alpha * grad
        grad_before_last = grad_before
        grad_before = grad
    return est


def derLA(X, Y, Z, x0_0, y0_0, A_0, sigma_0):
    result = 0
    for x_i, y_i, z in zip(X, Y, Z):
        result += (F1(x_i, y_i, A_0, sigma_0, x0_0, y0_0) - z) * \
                  F1_diff_A.evalf(subs={x: x_i, y: y_i, A: A_0, sigma: sigma_0, x0: x0_0, y0: y0_0})
    result *= 2
    return result


def derLSigma(X, Y, Z, x0_0, y0_0, A_0, sigma_0):
    result = 0
    for x_i, y_i, z in zip(X, Y, Z):
        result += (F1(x_i, y_i, A_0, sigma_0, x0_0, y0_0) - z) * \
                  F1_diff_sigma.evalf(subs={x: x_i, y: y_i, A: A_0, sigma: sigma_0, x0: x0_0, y0: y0_0})
    result *= 2
    return result


def derLX0(X, Y, Z, x0_0, y0_0, A_0, sigma_0):
    result = 0
    for x_i, y_i, z in zip(X, Y, Z):
        result += (F1(x_i, y_i, A_0, sigma_0, x0_0, y0_0) - z) * \
                  F1_diff_x0.evalf(subs={x: x_i, y: y_i, A: A_0, sigma: sigma_0, x0: x0_0, y0: y0_0})
    result *= 2
    return result


def derLY0(X, Y, Z, x0_0, y0_0, A_0, sigma_0):
    result = 0
    for x_i, y_i, z in zip(X, Y, Z):
        result += (F1(x_i, y_i, A_0, sigma_0, x0_0, y0_0) - z) * \
                  F1_diff_y0.evalf(subs={x: x_i, y: y_i, A: A_0, sigma: sigma_0, x0: x0_0, y0: y0_0})
    result *= 2
    return result


def gradL(X, Y, Z, A_num, sigma_num, x0_num, y0_num,):
    return np.array([derLA(X, Y, Z, x0_num, y0_num, A_num, sigma_num),
                     derLSigma(X, Y, Z, x0_num, y0_num, A_num, sigma_num),
                     derLX0(X, Y, Z, x0_num, y0_num, A_num, sigma_num),
                     derLY0(X, Y, Z, x0_num, y0_num, A_num, sigma_num)], dtype=np.float64)


def F1(x_0, y_0, A_0, sigma_0, x0_0, y0_0):
    return F1_func.evalf(subs={x: x_0, y: y_0, A: A_0, sigma: sigma_0, x0: x0_0, y0: y0_0})


def findMSE(X, Y, Z, A_0, sigma_0, x0_0, y0_0):
    squad_error = 0
    N = Z.shape[0]
    for i in range(N):
        squad_error += (Z[i] - F1(X[i], Y[i], A_0, sigma_0, x0_0, y0_0)) ** 2
    return squad_error / N
