from commons.neural_network import TwoLayerNet
from datasets.mnist import load_mnist
import matplotlib.pyplot as plt
import numpy as np

(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, one_hot_label=True
)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# hyperparameters
iters_num = 10000  # set loops
train_size = x_train.shape[0]
batch_size = 100  # size of mini batch
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# loops for 1 epoch
iter_per_epoch = max(train_size / batch_size, 1)
for i in range(iters_num):
    # get mini batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # calculate gradient
    grad = network.gradient(x_batch, t_batch)

    # renewal parameters
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # write learning datas
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # calculate accuracy of 1 epoch
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('train acc, test acc | ' + str(train_acc) + ', ' + str(test_acc))

# draw graph
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
