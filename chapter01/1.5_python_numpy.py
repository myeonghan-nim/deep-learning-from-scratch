import numpy as np

# numpy list
x = np.array([1.0, 2.0, 3.0])
print('numpy:', x, type(x))

# numpy calculation
y = np.array([2.0, 4.0, 6.0])
print('calculate:', x + y, x - y, x * y, x / y, 'bootstrap:', x / 2)

# numpy nth array
A = np.array([[1, 2], [3, 4]])
print(A, A.shape, A.dtype, 'info of A')

B = np.array([[3, 0], [0, 6]])
print(A + B, 'plus')
print(A * B, 'multiple')
print(A * 10, 'broadcast')

# broadcast
# if size of mat A and B are diff, numpy match sizes with broadcast
C = np.array([10, 20])
print(A * C, 'broadcast')
