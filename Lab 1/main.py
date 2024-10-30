import numpy as np

matrix_array = np.zeros((64, 64), dtype=int)
center = (32, 32)
y, x = np.ogrid[:64, :64]
distance = np.sqrt((x - center[1])**2 + (y - center[0])**2)
matrix_array[distance <= 44] = 1

print(matrix_array)
