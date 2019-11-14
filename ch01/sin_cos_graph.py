# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# prepare data
x = np.arange(0, 6, 0.1)  # make range from 0 to 6 for 0.1
y1 = np.sin(x)
y2 = np.cos(x)

# draw graph
plt.plot(x, y1, label='sin')
plt.plot(x, y2, linestyle = '--', label='cos')  # draw cos function with a dotted line
plt.xlabel('x')  # name of x line
plt.ylabel('y')  # name of y line
plt.title('sin & cos')
plt.legend()
plt.show()
