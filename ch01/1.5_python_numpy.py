# 1. outer library, Numpy
import numpy as np


# 2. numpy list
x = np.array([1.0, 2.0, 3.0])
print(x)
print(type(x))


# 3. numpy list calculation
y = np.array([2.0, 4.0, 6.0])
print(x + y)
print(x - y)
print(x * y)
print(x / y)

print(x / 2)  # broadcast


# 4. numpy Nth array
A = np.array([[1, 2], [3, 4]])
print(A)
print(A.shape)
print(A.dtype)

B = np.array([[3, 0], [0, 6]])
print(A + B)
print(A * B)

print(A)
print(A * 10)  # broadcast


# 5. Broadcast

'''
when using numpy array
if size of A mat and B mat is diff
numpy match size of mat with broadcast
'''

C = np.array([10, 20])
print(A * C)
