import numpy as np
import matplotlib.pyplot as plt

from matplotlib.image import imread


# 1. matplotlib single graph
x = np.arange(0, 6, 0.1)
y = np.sin(x)  # sin

plt.plot(x, y)
plt.show()


# 2. matplotlib multi graph
a = np.arange(0, 6, 0.1)
b = np.sin(x)
c = np.cos(x)

plt.plot(a, b, label='sin')  # label makes plot label
plt.plot(a, c, linestyle='--', label='cos')  # set line style

plt.xlabel('number')  # name of x line
plt.ylabel('result')  # name of y line
plt.title('sin & cos')

plt.legend()
plt.show()


# 3. image showing with matplotlib
img = imread('../dataset/lena.png')  # load image
plt.imshow(img)

plt.show()
