import taichi as ti
import pickle
import tina


x = ti.Matrix.shield(3, 3, float, 4)
x[3][2, 1] = 1
print(x)
x = pickle.dumps(x)
print(x)
ti.init()
x = pickle.loads(x)
print(x)
exit(1)
