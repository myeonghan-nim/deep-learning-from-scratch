# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# prepare data
x = np.arange(0, 6, 0.1)  # make range from 0 to 6 for 0.1
y = np.sin(x)

# draw graph
plt.plot(x, y)
plt.show()
