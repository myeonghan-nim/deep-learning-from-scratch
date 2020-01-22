# 0. import
import numpy as np


# 1. make basic perceptron
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7

    y = x1 * w1 + x2 * w2
    if y > theta:
        return 1
    else:
        return 0


for i, j in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    print(AND(i, j))


# 2. advanced perceptron with weight bias
x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7  # theta

temp = x * w
print(np.sum(temp))
print(np.sum(temp) + b)


# 3. implement weight bias
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7  # theta

    y = np.sum(x * w) + b
    if y > 0:
        return 1
    else:
        return 0


for i, j in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    print(AND(i, j))


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7  # theta

    y = np.sum(x * w) + b
    if y > 0:
        return 1
    else:
        return 0


for i, j in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    print(NAND(i, j))


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2  # theta

    y = np.sum(x * w) + b
    if y > 0:
        return 1
    else:
        return 0


for i, j in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    print(OR(i, j))