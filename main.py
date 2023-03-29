import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from F1Approx import *
from F2Approx import *
from F3Approx import gridSearch, F3
from F2ApproxClass import F2ApproxClass
from F3ApproxSympy import gradientDescentF3
from FApprox import buildPlot

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
params = gradientDescentF1(x_array, y_array, z_array)
buildPlot(F1, params, x_array, y_array, z_array)


# Построение аппроксимирующей поверхности F2
f2_approx = F2ApproxClass()
params = f2_approx.approx(x_array, y_array, z_array)
print(params)
buildPlot(F2, params, x_array, y_array, z_array)


# Построение поверхности произвольной функции с помощью перебора по сетке
bottom_left_corner = np.array([0.3, 1500, 1500, -0.25, -0.25], dtype=np.float64)
top_right_corner = np.array([0.7, 2500, 2500, 0.25, 0.25], dtype=np.float64)
number_of_slice = 4
number_of_repeat = 50
cease_rate = 0.94
params = gridSearch(bottom_left_corner, top_right_corner, number_of_slice,
                    number_of_repeat, cease_rate, x_array, y_array, z_array)
buildPlot(F3, params, x_array, y_array, z_array)
