import numpy as np
from commons.gradient import numerical_gradient
from commons.functions import softmax, cross_entropy_error


class simpleNet:
    def __init__(self):  # init values with one-hot-encoind
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):  # calculate error
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss


x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()


def f(w):
    return net.loss(x, t)


dW = numerical_gradient(f, net.W)
print(dW)
