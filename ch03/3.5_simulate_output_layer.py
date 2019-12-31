# 1. softmax function
import numpy as np


def softmax(a):
    exp_a = np.exp(a)
    sum_a = np.sum(exp_a)

    return exp_a / sum_a


# 2. modified softmax function
def modified_softmax(a):
    maxA = np.max(a)

    exp_a = np.exp(a - maxA)
    sum_a = np.sum(exp_a)

    return exp_a / sum_a
