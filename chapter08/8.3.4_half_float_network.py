from datasets.mnist import load_mnist
from deep_convnet import DeepConvNet
import matplotlib.pyplot as plt
import numpy as np

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()
network.load_params('./chapter08/deep_convnet_params.pkl')

sampled = 10000  # samples for high speed
x_test = x_test[:sampled]
t_test = t_test[:sampled]

print('caluculate accuracy (float64) ...')
print(network.accuracy(x_test, t_test))

# type change to float16
x_test = x_test.astype(np.float16)
for param in network.params.values():
    param[...] = param.astype(np.float16)

print('caluculate accuracy (float16) ...')
print(network.accuracy(x_test, t_test))
