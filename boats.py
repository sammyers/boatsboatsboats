import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stl import mesh
import scipy.integrate as integrate
import sympy

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

x = np.linspace(-1.85, 0, 1000)
x2 = np.linspace(0, 1.85, 1000)
y = lambda x: np.tan(-(x+0.95)**3) + (x+0.95)**2 - 1.8
y2 = lambda x: np.tan((x+0.9)**3) + (-(x+0.9))**2 - 1.8
z = lambda x: np.exp(0.1*x**2)

# X, Z = np.meshgrid(x, y(x))
# Y = z(X)
# plt.plot(x, y(x))
# plt.ylim([-1.6, 0.3])
# plt.show()
# surf = ax.plot_surface(X, Y, Z, linewidth=0)

# xy = ax.plot(x, )
xz = ax.plot(x, np.linspace(0, 0, 1000), y(x))
xz2 = ax.plot(x2, np.linspace(0, 0, 1000), y2(x))

# print abs(integrate.quad(y, -1, 1)[0])
print y(0)
print y(-1.85)

plt.plot(x, z(x))
plt.show()

weighted_mass = lambda x, y: np.array([])