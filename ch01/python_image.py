# coding: utf-8
from matplotlib.image import imread
import matplotlib.pyplot as plt

img = imread('../dataset/lena.png')  # load image
plt.imshow(img)

plt.show()
