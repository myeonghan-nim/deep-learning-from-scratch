# 0. old import method
# import sys
# import os

# sys.path.append(os.pardir)

# from common.functions import sigmoid, softmax
# from dataset.mnist import load_mnist

# 1. new import method
from functions import sigmoid, softmax
from mnist import load_mnist
import pickle
import numpy as np


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=False
    )
    return x_test, t_test


def init_network():
    with open('ch03/3.x_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

batch_size = 100  # define size of batch
accuracy_cnt = 0

for i in range(0, len(x), batch_size):  # start checking images with batch step
    x_batch = x[i:i + batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)  # find max probability of array with axis
    accuracy_cnt += np.sum(p == t[i:i + batch_size])

print('Accuracy:' + str(float(accuracy_cnt) / len(x)))
