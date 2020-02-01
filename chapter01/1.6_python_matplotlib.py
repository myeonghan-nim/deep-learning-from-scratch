from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np


# matplotlib single graph
x = np.arange(0, 6, 0.1)
y = np.sin(x)

plt.plot(x, y)
plt.show()

# matplotlib multi graph
a = np.arange(0, 6, 0.1)
b = np.sin(x)
c = np.cos(x)

plt.plot(a, b, label='sin')  # plot label
plt.plot(a, c, linestyle='--', label='cos')  # line style

plt.xlabel('number')  # name of x
plt.ylabel('result')  # name of y
plt.title('sin & cos')

plt.legend()
plt.show()

# image showing with matplotlib
img = imread('./README.assets/thumb-course-phthon-basic-1573569963444.jpg')
plt.imshow(img)

plt.show()
