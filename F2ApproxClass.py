import math
import numpy as np


class F2ApproxClass:
    def __init__(self):
        self.min_mse = None

    def approx(self, X, Y, Z):
        return self._gradientDescentF2(X, Y, Z)

    def F2(self, A, sigma_x, sigma_y, x0, y0, x, y):
        return A * self._doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y)

    @staticmethod
    def _stepChange(alpha, grad, grad_before, grad_before_last):
        was_change = False
        for i in range(grad.shape[0]):
            if grad[i] > grad_before[i] and grad_before[i] < grad_before_last[i]:
                alpha[i] = alpha[i] * 0.5
                was_change = True
        return was_change

    def _gradientDescentF2(self, X, Y, Z):
        alpha = np.array([0.17, 10000, 10000, 0.2, 0.2])
        est = np.array([Z[11], 100, 100, 0., 0.])  # Start estimation
        grad_before_last = est
        grad_before = est
        for i in range(5001):
            grad = self._gradL(X, Y, Z, est[0], est[1], est[2], est[3], est[4])
            if self._stepChange(alpha, grad, grad_before, grad_before_last):
                grad = self._gradL(X, Y, Z, est[0], est[1], est[2], est[3], est[4])

            if i % 5000 == 0:
                print(self._findMSE(X, Y, Z, est[0], est[1], est[2], est[3], est[4]))

            est -= alpha * grad
            grad_before_last = grad_before
            grad_before = grad
        self.min_mse = self._findMSE(X, Y, Z, est[0], est[1], est[2], est[3], est[4])
        return est

    def _gradL(self, X, Y, Z, A, sigma_x, sigma_y, x0, y0):
        return np.array([self._derLA(X, Y, Z, x0, y0, A, sigma_x, sigma_y),
                         self._derLSigmaX(X, Y, Z, x0, y0, A, sigma_x, sigma_y),
                         self._derLSigmaY(X, Y, Z, x0, y0, A, sigma_x, sigma_y),
                         self._derLX0(X, Y, Z, x0, y0, A, sigma_x, sigma_y),
                         self._derLY0(X, Y, Z, x0, y0, A, sigma_x, sigma_y)], dtype=np.float64)

    @staticmethod
    def _gaussExp(x, x0, sigma):
        return math.exp((-(x - x0) ** 2) / (2 * sigma))

    def _doubleGaussExp(self, x, x0, y, y0, sigma_x, sigma_y):
        return self._gaussExp(x, x0, sigma_x) * self._gaussExp(y, y0, sigma_y)

    def _derF2A(self, x, y, x0, y0, sigma_x, sigma_y):
        return self._doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y)

    def _derF2X0(self, x, y, x0, y0, A, sigma_x, sigma_y):
        return A * self._doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y) * (x - x0) / sigma_x

    def _derF2Y0(self, x, y, x0, y0, A, sigma_x, sigma_y):
        return A * self._doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y) * (y - y0) / sigma_y

    def _derF2SigmaX(self, x, y, x0, y0, A, sigma_x, sigma_y):
        return A * self._doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y) * ((x - x0) ** 2 * 2 / ((2 * sigma_x) ** 2))

    def _derF2SigmaY(self, x, y, x0, y0, A, sigma_x, sigma_y):
        return A * self._doubleGaussExp(x, x0, y, y0, sigma_x, sigma_y) * ((y - y0) ** 2 * 2 / ((2 * sigma_y) ** 2))

    def _derLA(self, X, Y, Z, x0, y0, A, sigma_x, sigma_y):
        result = 0
        for x, y, z in zip(X, Y, Z):
            result += (self.F2(A, sigma_x, sigma_y, x0, y0, x, y) - z) * self._derF2A(x, y, x0, y0, sigma_x, sigma_y)
        result *= 2
        return result

    def _derLX0(self, X, Y, Z, x0, y0, A, sigma_x, sigma_y):
        result = 0
        for x, y, z in zip(X, Y, Z):
            result += (self.F2(A, sigma_x, sigma_y, x0, y0, x, y) - z) \
                      * self._derF2X0(x, y, x0, y0, A, sigma_x, sigma_y)
        result *= 2
        return result

    def _derLY0(self, X, Y, Z, x0, y0, A, sigma_x, sigma_y):
        result = 0
        for x, y, z in zip(X, Y, Z):
            result += (self.F2(A, sigma_x, sigma_y, x0, y0, x, y) - z) \
                      * self._derF2Y0(x, y, x0, y0, A, sigma_x, sigma_y)
        result *= 2
        return result

    def _derLSigmaX(self, X, Y, Z, x0, y0, A, sigma_x, sigma_y):
        result = 0
        for x, y, z in zip(X, Y, Z):
            result += (self.F2(A, sigma_x, sigma_y, x0, y0, x, y) - z) \
                      * self._derF2SigmaX(x, y, x0, y0, A, sigma_x, sigma_y)
        result *= 2
        return result

    def _derLSigmaY(self, X, Y, Z, x0, y0, A, sigma_x, sigma_y):
        result = 0
        for x, y, z in zip(X, Y, Z):
            result += (self.F2(A, sigma_x, sigma_y, x0, y0, x, y) - z) \
                      * self._derF2SigmaY(x, y, x0, y0, A, sigma_x, sigma_y)
        result *= 2
        return result

    def _findMSE(self, X, Y, Z, A, sigma_x, sigma_y, x0, y0):
        squad_error = 0
        N = Z.shape[0]
        for i in range(N):
            squad_error += (Z[i] - self.F2(A, sigma_x, sigma_y, x0, y0, X[i], Y[i])) ** 2
        return squad_error / N
