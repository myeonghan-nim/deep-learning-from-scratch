import numpy as np


def softmax(a):  # softmax function
    exp_a = np.exp(a)
    sum_a = np.sum(exp_a)

    return exp_a / sum_a


def modified_softmax(a):  # modified softmax function
    maxA = np.max(a)

    exp_a = np.exp(a - maxA)
    sum_a = np.sum(exp_a)

    return exp_a / sum_a
