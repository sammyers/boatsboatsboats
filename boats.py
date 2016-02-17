import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
import scipy.integrate as integrate
import sympy

x = np.linspace(-1, 1, 1000)
# y = np.abs(x)**3 - 1
y = lambda x: np.abs(x)**3 - 1
plt.plot(x, y(x))
plt.show()
print abs(integrate.quad(y, -1, 1)[0])