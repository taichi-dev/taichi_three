import taichi as ti
import numpy as np

ti.init(print_preprocessed=True)

@ti.kernel
def func(b: ti.ext_arr()):
    b.shape[0] + 4

func(np.array([1, 2]))