from .gradient import numerical_gradient
from .layers import *
from collections import OrderedDict
import numpy as np


class MultiLayerNet:
    '''
    full joint multi layer extended

    Parameters
    ----------
    input_size: input size(MNIST is 784)
    hidden_size_list: number of neurons list of eaxh layer
    output_size: output size(MNIST is 10)
    activation: activation function(relu or sigmoid)
    weight_init_std: standard deviation of weight
        relu or he: He init value
        sigmoid or'xavier: Xavier init value
    weight_decay_lambda: power of weight reduction
    '''

    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        self.__init_weight(weight_init_std)  # init weight

        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}  # create layer
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers['Affine' + str(idx)
                        ] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)
                        ] = activation_layer[activation]()

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)
                    ] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        '''
        init weight

        Parameters
        ----------
        weight_init_std: standard deviation of weight
            relu or he: He init value
            sigmoid or'xavier: Xavier init value
        '''

        all_size_list = \
            [self.input_size] + self.hidden_size_list + [self.output_size]

        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])

            self.params['W' + str(idx)] = scale * \
                np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        '''
        calculate loss function

        Parameters
        ----------
        x: input data
        t: answer label
        '''
        y = self.predict(x)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)

        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        '''
        calculate grad with numerical differential

        Parameters
        ----------
        x: input data
        t: answer label

        Returns
        -------
        grads of each layers dict
            grads['W1']、grads['W2']、...: weight of each layer
            grads['b1']、grads['b2']、...: bias of each layer
        '''
        def loss_W(W):
            return self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = numerical_gradient(loss_W,
                                                       self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W,
                                                       self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        '''
        calculate grad with backpropagation

        Parameters
        ----------
        x: input data
        t: answer label

        Returns
        -------
        grads of each layers dict
            grads['W1']、grads['W2']、...: weight of each layer
            grads['b1']、grads['b2']、...: bias of each layer
        '''

        self.loss(x, t)  # forward

        dout = 1  # backward
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}  # save result
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + \
                self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads
