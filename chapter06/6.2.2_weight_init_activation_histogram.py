import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


input_data = np.random.randn(1000, 100)  # 1000 datas
node_num = 100  # neural number of each layer
hidden_layer_size = 5  # 5 layers
activations = {}  # save results

x = input_data
for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # unlock init value which wants
    w = np.random.randn(node_num, node_num) * 1
    # w = np.random.randn(node_num, node_num) * 0.01
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    # w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)

    a = np.dot(x, w)

    # unlock activation function which wants
    z = sigmoid(a)
    # z = ReLU(a)
    # z = tanh(a)

    activations[i] = z

# draw histogram
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i + 1) + '-layer')

    if i != 0:
        plt.yticks([], [])

    plt.xlim(0.1, 1)
    plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0, 1))

plt.show()
