import matplotlib as mpl
import numpy as np
import pandas as pd

from F1ApproxClass import F1ApproxClass
from F2ApproxClass import F2ApproxClass
from F3ApproxClass import F3ApproxClass

mpl.use('TkAgg')  # for graphics correct work in PyCharm

x_array = pd.array([0, 0, 0, -30, -10, 10, 30, -10, 10, -30, -10, 10, 30, -10, 10, -30, -10, 10, 30, 0, 0, 0])
y_array = pd.array([100, 80, 60, 40, 40, 40, 40, 20, 20, 0, 0, 0, 0, -20, -20, -40, -40, -40, -40, -60, -80, -100])

f1_sum_MSE = 0
f2_sum_MSE = 0
f3_sum_MSE = 0
number_of_zonds = 11
for i in range(number_of_zonds):
    columns = ["z_" + str(i) for i in range(23)]
    df = pd.read_csv("zonds_data/zonds" + str(i + 1) + ".txt", sep="\t", names=columns, header=None)
    df.rename(columns={'z_0': 't'}, inplace=True)

    part_of_max_when_obs_starts = 1 / 5
    time_since_beam_starts = 15 * pow(10, -3)
    z_11_max_value = df['z_11'].max()
    beam_data = df[df.z_11 > part_of_max_when_obs_starts * z_11_max_value]
    observation_time = beam_data[beam_data.t > beam_data.head(1).t.values[0] + time_since_beam_starts].head(1)

    z_array = pd.array(observation_time.squeeze().to_list(), dtype=np.dtype("float32"))
    z_array = np.delete(z_array, 0)
    N = x_array.shape[0]

    intensity_table = pd.DataFrame({'x': x_array, 'y': y_array, 'z': z_array})

    number_of_repeat = 2000

    # Построение аппроксимации F1 функцией
    print(str(i) + " zonds F1 approximation")
    f1_approx = F1ApproxClass(number_of_repeat=number_of_repeat, start_est=np.array([z_array[11], 400, 0., 0.]),
                              start_alpha=np.array([0.1, 5000, 0.1, 0.1]), step_change_coef=0.8)
    f1_approx.gradientDescent(x_array, y_array, z_array)
    print("MSE: " + str(f1_approx.min_mse))
    f1_sum_MSE += f1_approx.min_mse

    # Построение аппроксимирующей поверхности F2
    print(str(i) + " zonds F2 approximation")
    f2_approx = F2ApproxClass(number_of_repeat=number_of_repeat, start_est=np.array([z_array[11], 100, 100, 0., 0.]),
                              start_alpha=np.array([0.17, 4000, 4000, 0.2, 0.2]), step_change_coef=0.8)
    f2_approx.gradientDescent(x_array, y_array, z_array)
    print("MSE: " + str(f2_approx.min_mse))
    f2_sum_MSE += f2_approx.min_mse

    # Построение аппроксимирующей поверхности F3
    print(str(i) + " zonds F3 approximation")
    f3_approx = F3ApproxClass(number_of_repeat=number_of_repeat, start_est=np.array([z_array[11], 100, 100, 0., 0.]),
                              start_alpha=np.array([0.17, 5000, 2500, 0.2, 0.2]), step_change_coef=0.8)
    f3_approx.gradientDescent(x_array, y_array, z_array)
    print("MSE: " + str(f3_approx.min_mse))
    f3_sum_MSE += f3_approx.min_mse
    print()

    if i == 8:
        f1_approx.buildPlot(x_array, y_array, z_array)
        f2_approx.buildPlot(x_array, y_array, z_array)
        f3_approx.buildPlot(x_array, y_array, z_array)

print("Mean MSE first function: " + str(f1_sum_MSE/number_of_zonds))
print("Mean MSE second function: " + str(f2_sum_MSE/number_of_zonds))
print("Mean MSE third function: " + str(f3_sum_MSE/number_of_zonds))
# You can make approx such way for every function you want
# bottom_left_corner = np.array([0.3, 1500, 1500, -0.25, -0.25], dtype=np.float64)
# top_right_corner = np.array([0.7, 2500, 2500, 0.25, 0.25], dtype=np.float64)
# number_of_slice = 4
# number_of_repeat = 50
# cease_rate = 0.94
# f2_approx.gridSearch(bottom_left_corner, top_right_corner, number_of_slice,
#                      number_of_repeat, cease_rate, x_array, y_array, z_array)
# f2_approx.buildPlot(x_array,y_array,z_array)
