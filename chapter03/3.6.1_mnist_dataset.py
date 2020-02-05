from datasets.mnist import load_mnist
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
img = img.reshape(28, 28)  # reshape 1d mat images
print(img.shape)

img_show(img)
