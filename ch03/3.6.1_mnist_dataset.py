# 0. original import method
# import sys
# import os

# sys.path.append(os.pardir)  # load files in parent dir

# from dataset.mnist import load_mnist

# 1. new import method
from mnist import load_mnist
from PIL import Image
import numpy as np


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=True)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)  # img file is 1st dimension array, so reshape it
print(img.shape)

img_show(img)
