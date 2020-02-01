import numpy as np

# multi-dimension matrix
A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A))  # check dimension
print(A.shape)  # check atoms
print(A.shape[0])

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
print(np.ndim(B))
print(B.shape)

# multiple of matrix
C = np.array([[1, 2], [3, 4]])
D = np.array([[5, 6], [7, 8]])
print(C.shape, D.shape)

E = np.dot(C, D)  # multiple with scala calculation
print(E)

# last dim of first mat must be same as first dim of next mat
F = np.array([[1, 2, 3], [4, 5, 6]])
G = np.array([[1, 2], [3, 4], [5, 6]])

H = np.dot(F, G)
print(H)

I = np.array([[1, 2], [3, 4], [5, 6]])
J = np.array([7, 8])

K = np.dot(I, J)
print(K)

# multiple of matrix in neural network
X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])

Y = np.dot(X, W)
print(Y)
