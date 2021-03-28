import taichi as ti
import numpy as np


A = np.array([
[0, 1, 0],
[1, 0, 1],
[0, 1, 0],
])


def conv(A, B):
    m, n = A.shape
    s, t = B.shape
    C = np.zeros((m + s - 1, n + t - 1))
    for i in range(m):
        for j in range(n):
            for k in range(s):
                for l in range(t):
                    C[i + k, j + l] += A[i, j] * B[k, l]
    return C


B = A
B = conv(B, A)
B = conv(B, A)
print(B)
