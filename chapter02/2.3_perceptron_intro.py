import numpy as np


def AND(x1, x2):  # AND perceptron
    w1, w2, theta = 0.5, 0.5, 0.7

    y = x1 * w1 + x2 * w2
    if y > theta:
        return 1
    else:
        return 0


print('AND:', end=' ')
for i, j in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    print(AND(i, j), end=' ')
print()

# advanced perceptron with weight bias
x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7  # theta

temp = x * w
print('bias calculation:', np.sum(temp), np.sum(temp) + b)


def AND(x1, x2):  # implement weight bias
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7

    y = np.sum(x * w) + b
    if y > 0:
        return 1
    else:
        return 0


print('advanced AND:', end=' ')
for i, j in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    print(AND(i, j), end=' ')
print()


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7

    y = np.sum(x * w) + b
    if y > 0:
        return 1
    else:
        return 0


print('NAND:', end=' ')
for i, j in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    print(NAND(i, j), end=' ')
print()


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2

    y = np.sum(x * w) + b
    if y > 0:
        return 1
    else:
        return 0


print('OR:', end=' ')
for i, j in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    print(OR(i, j), end=' ')
print()
