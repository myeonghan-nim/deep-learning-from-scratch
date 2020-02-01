import numpy as np


def sigmoid(x):  # sigmoid function
    return 1 / (1 + np.exp(-x))


def identity_function(x):  # return function for print
    return x


# 1st -> 2nd layer
X = np.array([1.0, 0.5])
W1 = np.array([
    [0.1, 0.3, 0.5],
    [0.2, 0.4, 0.6]
])
B1 = np.array([0.1, 0.2, 0.3])

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)
print(Z1)

# 2nd -> 3rd layer
W2 = np.array([
    [0.1, 0.4], [0.2, 0.5], [0.3, 0.6]
])
B2 = np.array([0.1, 0.2])

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print(Z2)

# 3rd -> return layer
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)
print(Y)


# summary multi layer neural network
def init_network():
    network = {}

    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])

    network['B1'] = np.array([0.1, 0.2, 0.3])
    network['B2'] = np.array([0.1, 0.2])
    network['B3'] = np.array([0.1, 0.2])

    return network


def foward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    B1, B2, B3 = network['B1'], network['B2'], network['B3']

    A1 = np.dot(x, W1) + B1
    Z1 = sigmoid(A1)
    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)
    A3 = np.dot(Z2, W3) + B3
    Y = identity_function(A3)

    return Y


network = init_network()
x = np.array([1.0, 0.5])
y = foward(network, x)
print(y)
