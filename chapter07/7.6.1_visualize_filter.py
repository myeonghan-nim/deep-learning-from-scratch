from cnn import SimpleConvNet
import matplotlib.pyplot as plt
import numpy as np


def filter_show(filters, nx=8, margin=3, scale=10):
    '''
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    '''
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0,
                        top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i + 1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')

    plt.show()


network = SimpleConvNet()  # bias after random init
filter_show(network.params['W1'])

network.load_params('chapter07/params.pkl')  # learned bias
filter_show(network.params['W1'])
