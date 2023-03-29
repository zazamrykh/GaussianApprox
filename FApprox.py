import math

import numpy as np
from matplotlib import pyplot as plt
from numpy import arange
from numpy import linalg as LA


def stepChange(alpha, grad, grad_before, grad_before_last):
    was_change = False
    for i in range(grad.shape[0]):
        if grad[i] > grad_before[i] and grad_before[i] < grad_before_last[i]:
            alpha[i] = alpha[i] * 0.5
            was_change = True
    return was_change


class FApprox:
    def __init__(self, number_of_repeat, start_est, start_alpha, F, gradL):
        self.start_est = start_est
        self.number_of_repeat = number_of_repeat
        self.start_alpha = start_alpha
        self.gradL = gradL
        self.F = F
        self.params = None
        self.min_mse = None

    # alpha = np.array([0.17, 10000, 10000, 0.2, 0.2])
    # est = np.array([Z[11], 100, 100, 0., 0.])  # Start estimation

    def gradientDescent(self, X, Y, Z):
        alpha = self.start_alpha
        est = self.start_est
        grad_before_last = est
        grad_before = est
        for i in range(self.number_of_repeat):
            grad = self.gradL(X, Y, Z, *est)
            if stepChange(alpha, grad, grad_before, grad_before_last):
                grad = self.gradL(X, Y, Z, *est)
            est -= alpha * grad
            grad_before_last = grad_before
            grad_before = grad
        self.min_mse = self.findMSE(X, Y, Z, *est)
        print(self.min_mse)
        self.params = est

    def gridSearch(self, bottom_left_corner, top_right_corner, number_of_slice, number_of_repeat, cease_rate, X, Y, Z):
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
                            current_mse = self.findMSE(X, Y, Z, current_a, current_sigma_x,
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
                                           best_array[4] - step_y0 * number_of_slice / 2 * cease_rate],
                                          dtype=np.float64)
            top_right_corner = np.array([best_array[0] + step_a * number_of_slice / 2 * cease_rate,
                                         best_array[1] + step_sigma_x * number_of_slice / 2 * cease_rate,
                                         best_array[2] + step_sigma_y * number_of_slice / 2 * cease_rate,
                                         best_array[3] + step_x0 * number_of_slice / 2 * cease_rate,
                                         best_array[4] + step_y0 * number_of_slice / 2 * cease_rate], dtype=np.float64)
            best_array = self.gridSearch(bottom_left_corner, top_right_corner, number_of_slice, number_of_repeat,
                                         cease_rate,
                                         X, Y, Z)
        return best_array

    def buildPlot(self, x_array, y_array, z_array):
        fig = plt.figure()
        fig.set_figheight(10)
        fig.set_figwidth(10)
        plt.gca().set_aspect('equal')
        ax = plt.subplot(projection='3d')
        ax.scatter(x_array, y_array, z_array, color='red')
        x_grid, y_grid = np.meshgrid(np.arange(-30, 30, 0.5), np.arange(-100, 100, 0.5))

        width = x_grid.shape[0]
        length = x_grid.shape[1]
        z_grid = np.zeros((width, length))

        for i in range(width):
            for j in range(length):
                z_grid[i][j] = self.F(*self.params, x_grid[i][j], y_grid[i][j])
        ax.plot_wireframe(x_grid, y_grid, z_grid)
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        plt.show()

    def findMSE(self, X, Y, Z, A, sigma_x, sigma_y, x0, y0):
        squad_error = 0
        N = Z.shape[0]
        for i in range(N):
            squad_error += (Z[i] - self.F(A, sigma_x, sigma_y, x0, y0, X[i], Y[i])) ** 2
        return squad_error / N
