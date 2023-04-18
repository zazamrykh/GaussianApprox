import matplotlib as mpl
import numpy as np
import pandas as pd

from F1ApproxClass import F1ApproxClass
from F2ApproxClass import F2ApproxClass
from F3ApproxClass import F3ApproxClass

columns = ["z_" + str(i) for i in range(23)]
df = pd.read_csv("VseZondy.txt", sep="\t", names=columns, header=None)
df.rename(columns={'z_0': 't'}, inplace=True)

part_of_max_when_obs_starts = 1 / 5
time_since_beam_starts = 15 * pow(10, -3)
z_11_max_value = df['z_11'].max()
beam_data = df[df.z_11 > part_of_max_when_obs_starts * z_11_max_value]
observation_time = beam_data[beam_data.t > beam_data.head(1).t.values[0] + time_since_beam_starts].head(1)

x_array = pd.array([0, 0, 0, -30, -10, 10, 30, -10, 10, -30, -10, 10, 30, -10, 10, -30, -10, 10, 30, 0, 0, 0])
y_array = pd.array([100, 80, 60, 40, 40, 40, 40, 20, 20, 0, 0, 0, 0, -20, -20, -40, -40, -40, -40, -60, -80, -100])
z_array = pd.array(observation_time.squeeze().to_list(), dtype=np.dtype("float32"))
z_array = np.delete(z_array, 0)
N = x_array.shape[0]

mpl.use('TkAgg')
intensity_table = pd.DataFrame({'x': x_array, 'y': y_array, 'z': z_array})

# Построение аппроксимации F1 функцией
print("F1 approximation")
f1_approx = F1ApproxClass(number_of_repeat=5000, start_est=np.array([z_array[11], 400, 0., 0.]),
                          start_alpha=np.array([0.1, 5000, 0.1, 0.1]))
f1_approx.gradientDescent(x_array, y_array, z_array)
print("Gotten parameters:")
print(f1_approx.params)
print("F1 MSE:")
print(f1_approx.min_mse)
f1_approx.buildPlot(x_array, y_array, z_array)
print()

# Построение аппроксимирующей поверхности F2
print("F2 approximation")
f2_approx = F2ApproxClass(number_of_repeat=5000, start_est=np.array([z_array[11], 100, 100, 0., 0.]),
                          start_alpha=np.array([0.17, 4000, 4000, 0.2, 0.2]))
f2_approx.gradientDescent(x_array, y_array, z_array)
print("Gotten parameters:")
print(f2_approx.params)
print("F2 MSE:")
print(f2_approx.min_mse)
f2_approx.buildPlot(x_array, y_array, z_array)
print()

# Построение аппроксимирующей поверхности F3
print("F3 approximation")
f3_approx = F3ApproxClass(number_of_repeat=5000, start_est=np.array([z_array[11], 100, 100, 0., 0.]),
                          start_alpha=np.array([0.17, 5000, 2500, 0.2, 0.2]), step_change_coef=0.8)
f3_approx.gradientDescent(x_array, y_array, z_array)
print("Gotten parameters:")
print(f3_approx.params)
print("F3 MSE:")
print(f3_approx.min_mse)
f3_approx.buildPlot(x_array, y_array, z_array)

# You can make approx such way for every function you want
# bottom_left_corner = np.array([0.3, 1500, 1500, -0.25, -0.25], dtype=np.float64)
# top_right_corner = np.array([0.7, 2500, 2500, 0.25, 0.25], dtype=np.float64)
# number_of_slice = 4
# number_of_repeat = 50
# cease_rate = 0.94
# f2_approx.gridSearch(bottom_left_corner, top_right_corner, number_of_slice,
#                      number_of_repeat, cease_rate, x_array, y_array, z_array)
# f2_approx.buildPlot(x_array,y_array,z_array)
