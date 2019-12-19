import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    # dtype changes True/False to 1/0 so it returns np.array with 1/0
    return np.array(x > 0, dtype=np.int)


# check each atom in X is over 0
X = np.arange(-5.0, 5.0, 0.1)
Y = step_function(X)

plt.plot(X, Y)
plt.ylim(-0.1, 1.1)
plt.show()
