import math

import numpy as np
from numpy import arange
from numpy import linalg as LA


def gridSearch(bottom_left_corner, top_right_corner, number_of_slice, number_of_repeat, cease_rate, X, Y, Z):
    step_a = (top_right_corner[0] - bottom_left_corner[0]) / number_of_slice
    step_sigma_x = (top_right_corner[1] - bottom_left_corner[1]) / number_of_slice
    step_sigma_y = (top_right_corner[2] - bottom_left_corner[2]) / number_of_slice
    step_x0 = (top_right_corner[3] - bottom_left_corner[3]) / number_of_slice
    step_y0 = (top_right_corner[4] - bottom_left_corner[4]) / number_of_slice
    best_mse = math.inf
    best_array = np.zeros(shape=(5, 1), dtype=np.float64)
    for current_a in arange(bottom_left_corner[0], top_right_corner[0], step_a):
        for current_sigma_x in arange(bottom_left_corner[1], top_right_corner[1], step_sigma_x):
            for current_sigma_y in arange(bottom_left_corner[2], top_right_corner[2], step_sigma_y):
                for current_x0 in arange(bottom_left_corner[3], top_right_corner[3], step_x0):
                    for current_y0 in arange(bottom_left_corner[4], top_right_corner[4], step_y0):
                        current_mse = findMSE(X, Y, Z, current_a, current_sigma_x,
                                              current_sigma_y, current_x0, current_y0)
                        if current_mse < best_mse:
                            best_mse = current_mse
                            best_array[0] = current_a
                            best_array[1] = current_sigma_x
                            best_array[2] = current_sigma_y
                            best_array[3] = current_x0
                            best_array[4] = current_y0
    print(number_of_repeat)
    print(best_array)
    print(best_mse)
    print(LA.norm(top_right_corner - bottom_left_corner))
    if number_of_repeat > 1:
        number_of_repeat -= 1
        bottom_left_corner = np.array([best_array[0] - step_a * number_of_slice / 2 * cease_rate,
                                       best_array[1] - step_sigma_x * number_of_slice / 2 * cease_rate,
                                       best_array[2] - step_sigma_y * number_of_slice / 2 * cease_rate,
                                       best_array[3] - step_x0 * number_of_slice / 2 * cease_rate,
                                       best_array[4] - step_y0 * number_of_slice / 2 * cease_rate], dtype=np.float64)
        top_right_corner = np.array([best_array[0] + step_a * number_of_slice / 2 * cease_rate,
                                     best_array[1] + step_sigma_x * number_of_slice / 2 * cease_rate,
                                     best_array[2] + step_sigma_y * number_of_slice / 2 * cease_rate,
                                     best_array[3] + step_x0 * number_of_slice / 2 * cease_rate,
                                     best_array[4] + step_y0 * number_of_slice / 2 * cease_rate], dtype=np.float64)
        best_array = gridSearch(bottom_left_corner, top_right_corner, number_of_slice, number_of_repeat, cease_rate,
                                X, Y, Z)
    return best_array


def F3(A, sigma_x, sigma_y, x0, y0, x, y):
    # return A * doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y)
    return A * doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y) * (
            1 + (((y - y0) ** 2) / (2 * sigma_y)) * gaussExp(y, y0, sigma_y))
    # return A * doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y) * (
    #         1 + (((y - y0) ** 2) / (2 * sigma_y)) * gaussExp(y, y0, sigma_y)) * (
    #         1 + (((x - x0) ** 2) / (2 * sigma_x)) * gaussExp(x, x0, sigma_x))


def gaussExp(x, x0, sigma):
    return math.exp((-(x - x0) ** 2) / (2 * sigma))


def doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y):
    return gaussExp(x, x0, sigma_x) * gaussExp(y, y0, sigma_y)


def findMSE(X, Y, Z, A, sigma_x, sigma_y, x0, y0):
    squad_error = 0
    N = Z.shape[0]
    for i in range(N):
        squad_error += (Z[i] - F3(A, sigma_x, sigma_y, x0, y0, X[i], Y[i])) ** 2
    return squad_error / N

# def stepChange(alpha, grad, grad_before, grad_before_last):
#     was_change = False
#     for i in range(grad.shape[0]):
#         if grad[i] > grad_before[i] and grad_before[i] < grad_before_last[i]:
#             alpha[i] = alpha[i] * 0.5
#             was_change = True
#     return was_change
#
#
# def gradientDescentF3(X, Y, Z):
#     alpha = np.array([0.17, 10000, 10000, 0.2, 0.2])
#     est = np.array([Z[11], 100, 100, 0., 0.])  # Start estimation
#     grad_before_last = est
#     grad_before = est
#     for i in range(50001):
#         grad = gradL(X, Y, Z, est[0], est[1], est[2], est[3], est[4])
#         if stepChange(alpha, grad, grad_before, grad_before_last):
#             grad = gradL(X, Y, Z, est[0], est[1], est[2], est[3], est[4])
#         # print("Step number " + str(i) + " alpha value: " + str(alpha) + "\n" + str(grad))
#
#         if i % 5000 == 0:
#             print(findMSE(X, Y, Z, est[0], est[1], est[2], est[3], est[4]))
#
#         est -= alpha * grad
#         grad_before_last = grad_before
#         grad_before = grad
#     return est
#
#
# def gradL(X, Y, Z, A, sigma_x, sigma_y, x0, y0):
#     return np.array([derLA(X, Y, Z, x0, y0, A, sigma_x, sigma_y),
#                      derLSigmaX(X, Y, Z, x0, y0, A, sigma_x, sigma_y),
#                      derLSigmaY(X, Y, Z, x0, y0, A, sigma_x, sigma_y),
#                      derLX0(X, Y, Z, x0, y0, A, sigma_x, sigma_y),
#                      derLY0(X, Y, Z, x0, y0, A, sigma_x, sigma_y)], dtype=np.float64)
#
#
#
#
# def derF3A(x, y, x0, y0, sigma_x, sigma_y):
#     return doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y) * (
#             1 + ((y - y0) ** 2 / (2 * sigma_y)) * gaussExp(y, y0, sigma_y))
#
#
# def derF3X0(x, y, x0, y0, A, sigma_x, sigma_y):
#     return A * doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y) * (x - x0) / sigma_x
#
#
# def derF3Y0(x, y, x0, y0, A, sigma_x, sigma_y):
#     return A * doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y) * (y - y0) / sigma_y
#
#
# def derF3SigmaX(x, y, x0, y0, A, sigma_x, sigma_y):
#     return A * doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y) * ((x - x0) ** 2 * 2 / ((2 * sigma_x) ** 2))
#
#
# def derF3SigmaY(x, y, x0, y0, A, sigma_x, sigma_y):
#     return A * doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y) * ((y - y0) ** 2 * 2 / ((2 * sigma_y) ** 2))
#
#
# def derLA(X, Y, Z, x0, y0, A, sigma_x, sigma_y):
#     result = 0
#     for x, y, z in zip(X, Y, Z):
#         result += (F2(x, x0, y, y0, A, sigma_x, sigma_y) - z) * derF2A(x, y, x0, y0, sigma_x, sigma_y)
#     result *= 2
#     return result
#
#
# def derLX0(X, Y, Z, x0, y0, A, sigma_x, sigma_y):
#     result = 0
#     for x, y, z in zip(X, Y, Z):
#         result += (F2(x, x0, y, y0, A, sigma_x, sigma_y) - z) * derF2X0(x, y, x0, y0, A, sigma_x, sigma_y)
#     result *= 2
#     return result
#
#
# def derLY0(X, Y, Z, x0, y0, A, sigma_x, sigma_y):
#     result = 0
#     for x, y, z in zip(X, Y, Z):
#         result += (F2(x, x0, y, y0, A, sigma_x, sigma_y) - z) * derF2Y0(x, y, x0, y0, A, sigma_x, sigma_y)
#     result *= 2
#     return result
#
#
# def derLSigmaX(X, Y, Z, x0, y0, A, sigma_x, sigma_y):
#     result = 0
#     for x, y, z in zip(X, Y, Z):
#         result += (F2(x, x0, y, y0, A, sigma_x, sigma_y) - z) * derF2SigmaX(x, y, x0, y0, A, sigma_x, sigma_y)
#     result *= 2
#     return result
#
#
# def derLSigmaY(X, Y, Z, x0, y0, A, sigma_x, sigma_y):
#     result = 0
#     for x, y, z in zip(X, Y, Z):
#         result += (F2(x, x0, y, y0, A, sigma_x, sigma_y) - z) * derF2SigmaY(x, y, x0, y0, A, sigma_x, sigma_y)
#     result *= 2
#     return result
#
#
