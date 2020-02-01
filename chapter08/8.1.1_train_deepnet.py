from datasets.mnist import load_mnist
from commons.trainer import Trainer
from deep_convnet import DeepConvNet
import matplotlib.pyplot as plt
import numpy as np

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# save parameters
network.save_params('./chapter08/deep_convnet_params.pkl')
print('Saved Network Parameters!')
