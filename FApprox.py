import numpy as np
from matplotlib import pyplot as plt

def buildPlot(F, params, x_array, y_array, z_array):
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
            z_grid[i][j] = F(*params, x_grid[i][j], y_grid[i][j])
    ax.plot_wireframe(x_grid, y_grid, z_grid)
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.show()

