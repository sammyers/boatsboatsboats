import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import fsolve

fig = plt.figure()

xz = fig.add_subplot(111)

xz.set_xlim(-1, 1)
xz.set_ylim(-1.5, 0.5)

n = 3
theta = 72
deck_level = 0

def hull_xz(x, n=3):
	return abs(x)**n - 1

def waterline_xz(x, d, theta=theta):
	radians = np.radians(theta)
	return np.tan(radians) * x - d

def deck(x):
	return x - x

def boat_area():
	return integrate.dblquad(lambda x, y: 1, -1, 1, hull_xz, deck)[0]

def water_area(d):
	waterline1 = fsolve(lambda x: hull_xz(x) - waterline_xz(x, d), -1)[0]
	waterline2 = fsolve(lambda x: hull_xz(x) - waterline_xz(x, d), 1)[0]
	if waterline2 <= 1 and waterline1 != waterline2:
		integral = integrate.dblquad(lambda x, y: 1, waterline1, waterline2, hull_xz, lambda x: waterline_xz(x, d))[0]
		return integral
	else:
		waterline3 = fsolve(lambda x: deck(x) - waterline_xz(x, d), 0)[0]
		if not -1 <= waterline1 <= 1:
			waterline1 = fsolve(lambda x: hull_xz(x) - waterline_xz(x, d), 0)[0]
			print 'waterline1: ' + str(waterline1)
			print 'waterline3: ' + str(waterline3)
		integral1 = integrate.dblquad(lambda x, y: 1, waterline1, waterline3, hull_xz, lambda x: waterline_xz(x, d))[0]
		integral2 = integrate.dblquad(lambda x, y: 1, waterline3, 1, hull_xz, deck)[0]
		print integral1, integral2
		return integral1 + integral2

level = fsolve(lambda x: water_area(x) - 0.1, 1)[0]
print 'level: ' + str(level)

x = np.linspace(-1, 1, 100)
z = hull_xz(x)
w = waterline_xz(x, level)

if __name__ == "__main__":

	print 'area: ' + str(boat_area())
	print 'water area: ' + str(water_area(level))

	xz.plot(x, z, 'r')
	xz.plot(x, deck(x), 'r')
	xz.plot(x, w, 'b')

	plt.show()